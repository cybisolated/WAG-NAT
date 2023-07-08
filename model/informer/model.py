import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from ..embeddings import *
from dataset import BaseDataset


class Informer(nn.Module, BaseDataset):
    def __init__(
        self,
        L_in,
        L_token,
        L_out,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn='prob',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        num_classes=[],
        target=[],
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        time_varying_unknown_reals=[],
        time_varying_unknown_categoricals=[],
        static_reals=[],
        static_categoricals=[],
        **kwargs
    ):
        super(Informer, self).__init__()

        self.target = target
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.static_reals = static_reals
        self.static_categoricals = static_categoricals
        self.L_in = L_in
        self.L_out = L_out
        self.L_token = L_token

        self.attn = attn
        self.output_attention = output_attention

        encoder_real_size = len(self.encoder_reals) + len(self.targets)
        encoder_num_classes = []
        token_num_classes = []
        for name in self.encoder_cats:
            encoder_num_classes.append(num_classes[name])
            token_num_classes.append(num_classes[name])

        pred_num_classes = []
        for name in self.decoder_cats:
            pred_num_classes.append(num_classes[name])

        output_size = len(self.targets)

        # Embedding
        self.enc_embedding = DataEmbedding(encoder_real_size, encoder_num_classes, d_model, dropout, pos_type='sin')
        self.dec_embedding = DataEmbedding(encoder_real_size, encoder_num_classes, d_model, dropout, pos_type='sin')

        self.dec_real_embedding = RealEmbedding(encoder_real_size, d_model)
        self.dec_cat_embedding = nn.ModuleDict(
            {
                'token': CategoricalEmbedding(token_num_classes, d_model),
                'pred': CategoricalEmbedding(pred_num_classes, d_model),
            }
        )
        self.dec_pos_embedding = SinusoidalPositionEmbedding(d_model)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, output_size, bias=True)

    def forward(
        self,
        x,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # encoder
        x_encoder_reals = torch.cat([x['encoder_reals'], x['encoder_target']], dim=-1)
        x_encoder_cats = x['encoder_cats']
        enc_input = self.enc_embedding(x_encoder_reals, x_encoder_cats)
        enc_out, attns = self.encoder(enc_input, attn_mask=enc_self_mask)

        # decoder:
        # start token部分：
        # 相当于encoder的后面一部分

        # 预测部分：
        # 需要和start token的维度对齐。
        # 对于实数特征，和start token的实数特征维度对齐，future unknown的部分填充为0，在时间维度上拼接后共享一个Embedding
        # 对于类别特征，start token和预测部分的类别特征分开Embedding，然后在时间维度进行拼接，最后和实数的Embedding相加
        ## start token
        x_token_reals = x_encoder_reals[:, -self.L_token :, :]
        x_token_cats = x_encoder_cats[:, -self.L_token :, :]

        x_decoder_reals = x['decoder_reals']
        x_decoder_reals = torch.cat(
            [
                x_decoder_reals,
                torch.zeros(
                    *x_decoder_reals.shape[:2],
                    len(self.time_varying_unknown_reals + self.targets),
                    device=x_decoder_reals.device
                ),
            ],
            dim=-1,
        )
        ## 在时间维度上拼接两者的实数部分
        x_decoder_reals = torch.cat([x_token_reals, x_decoder_reals], dim=1)
        dec_real_input = self.dec_real_embedding(x_decoder_reals)

        x_decoder_cats = x['decoder_cats']
        dec_cat_input = (
            torch.cat(
                [self.dec_cat_embedding['token'](x_token_cats), self.dec_cat_embedding['pred'](x_decoder_cats)], dim=1
            )
            if x_token_cats.shape[-1] != 0
            else 0.0
        )

        dec_input = dec_real_input + dec_cat_input
        dec_input = dec_input + self.dec_pos_embedding(dec_input)

        dec_out = self.decoder(dec_input, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        out = self.projection(dec_out)

        if self.output_attention:
            return out[:, -self.L_out :, :], attns
        else:
            return out[:, -self.L_out :, :]  # [B, L_out, D]


class InformerStack(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=[3, 2, 1],
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn='prob',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        device=torch.device('cuda:0'),
    ):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                            ),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(el)
                ],
                [ConvLayer(d_model) for l in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model),
            )
            for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
