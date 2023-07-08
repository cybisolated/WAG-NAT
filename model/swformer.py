import torch
from torch import nn
from dataset.base import BaseDataset
from .decoder import Decoder
from .embeddings import *
from .encoder import Encoder, CanonicalEncoder
from .generator import QueryGenerator
from .sub_models import *


class SWFormer(nn.Module, BaseDataset):
    def __init__(
        self,
        L_in,
        L_out,
        num_classes,
        d_model,
        window_size,
        kernel_size,
        num_heads,
        num_enc_layers,
        num_dec_layers,
        activation,
        dropout,
        encoder_type,
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

        ffn_hidden_size = d_model * 4
        self.target = target
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.static_reals = static_reals
        self.static_categoricals = static_categoricals
        self.L_in = L_in
        self.L_out = L_out
        self.enc_attns = None

        encoder_real_size = len(self.encoder_reals) + len(self.targets)
        encoder_num_classes = []
        for name in self.encoder_cats:
            encoder_num_classes.append(num_classes[name])

        generator_real_size = len(self.decoder_reals)
        generator_num_classes = []
        for name in self.decoder_cats:
            generator_num_classes.append(num_classes[name])

        self.output_size = len(self.targets)

        self.encoder_embedding = DataEmbedding(encoder_real_size, encoder_num_classes, d_model, dropout, pos_type='sin')
        if encoder_type == 'window':
            self.encoder = Encoder(
                d_model, window_size, kernel_size, num_heads, num_enc_layers, ffn_hidden_size, activation, dropout
            )
        elif encoder_type == 'canonical':
            self.encoder = CanonicalEncoder(d_model, num_heads, num_enc_layers, ffn_hidden_size, activation, dropout)

        self.generator_embedding = DataEmbedding(
            generator_real_size, generator_num_classes, d_model, dropout, pos_type='none'
        )
        self.query_generator = QueryGenerator(L_out, d_model, num_heads, dropout)

        # self.decoder_embedding = nn.Linear(decoder_input_size, d_model)
        # self.decoder_pe = SinusoidalPositionEmbedding(d_model)
        self.decoder = Decoder(d_model, num_heads, num_dec_layers, ffn_hidden_size, activation, dropout)

        self.proj_layer = nn.Linear(d_model, self.output_size)

    def encode(self, enc_input):
        return self.encoder(enc_input)

    def generate_queries(self, gen_input, enc_input, enc_out, gen_mask=None):
        queries = self.query_generator(gen_input, enc_input, enc_out, gen_mask)
        return queries

    def decode(self, queries, encoder_out, src_mask=None, tgt_mask=None):
        # queries = self.decoder_pe(queries) + queries
        return self.decoder(queries, encoder_out, src_mask, tgt_mask)

    def forward(self, x):
        x_encoder_reals = torch.cat([x['encoder_reals'], x['encoder_target']], dim=-1)
        x_encoder_cats = x['encoder_cats']
        enc_input = self.encoder_embedding(x_encoder_reals, x_encoder_cats)

        x_decoder_reals = x['decoder_reals']
        x_decoder_cats = x['decoder_cats']
        gen_input = self.generator_embedding(x_decoder_reals, x_decoder_cats)

        encoder_out, self.enc_attns = self.encode(enc_input)

        gen_mask = None
        queries = self.generate_queries(gen_input, enc_input, encoder_out, gen_mask)

        # generate target mask
        # * decoder do not need mask to prevent accessing unseeable information
        # * here only mask itself to prevent attenting to itself to improve performance
        batch_size = x_encoder_reals.shape[0]
        tgt_mask = torch.ones(self.L_out, self.L_out, device=x_decoder_reals.device)
        diag_idx = torch.arange(self.L_out, device=x_decoder_reals.device)
        tgt_mask[diag_idx, diag_idx] = 0
        tgt_mask = tgt_mask.expand(batch_size, *tgt_mask.shape)
        # print(tgt_mask.shape, queries.shape)

        decoder_out = self.decode(queries, encoder_out, tgt_mask=tgt_mask)

        out = self.proj_layer(decoder_out)

        return out
