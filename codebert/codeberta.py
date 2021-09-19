# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from torch.nn import Module
from transformers import RobertaConfig, RobertaModel, GPT2Config, GPT2LMHeadModel
from transformers import EncoderDecoderConfig, EncoderDecoderModel
import torch.nn.functional as F
import numpy as np


class CodeBERTa(Module):
    def __init__(
        self, hidden_size, out_context_size, 
        max_position_embeddings, vocab_size, verbose=True
    ):
        super(CodeBERTa, self).__init__()

        num_hidden_layers = int(
            np.ceil(np.log(hidden_size + 0.) / np.log(3 + 0.)))
        intermediate_size = hidden_size  # 4 * hidden_size
        num_attention_heads = max(1, int(hidden_size // 64))

        encoder_config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings
        )

        decoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=2,
            n_head=num_attention_heads,
            n_ctx=out_context_size,
            n_positions=out_context_size
        )

        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config, decoder_config)
        self.model = EncoderDecoderModel(config=config)

        self.model.config.decoder.is_decoder = True
        self.model.config.decoder.add_cross_attention = True

        if verbose:
            dec_sz = GPT2LMHeadModel(decoder_config).num_parameters()
            enc_sz = RobertaModel(encoder_config).num_parameters()

            print('ENC', '%.1E' % enc_sz)
            print('DEC', '%.1E' % dec_sz)
            print('total', '%.1E' % (enc_sz + dec_sz))

    def forward(self, x, l, decoder_attention_mask=None):
        return self.model(
            labels=l,
            input_ids=x,
            decoder_input_ids=l*decoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )

    @staticmethod
    def loss_fn(outputs, targets, batch):
        outputs = outputs.permute(0, 2, 1)
        targets = targets.permute(0, 1)
        loss = F.cross_entropy(outputs, targets, reduction="none")
        mask = targets != 1
        loss = loss * mask
        return loss.sum() / batch
