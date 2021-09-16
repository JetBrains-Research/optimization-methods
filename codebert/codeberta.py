# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from torch.nn import Module
from transformers import RobertaModel, OpenAIGPTConfig, GPT2Config, OpenAIGPTLMHeadModel, GPT2LMHeadModel, RobertaConfig
from transformers import EncoderDecoderConfig, EncoderDecoderModel, BertLMHeadModel, BertConfig
import torch.nn.functional as F
import numpy as np


class CodeBERTa(Module):
    def __init__(self, hidden_size=192, context_size=80, max_position_embeddings=256, vocab_size=10_000, roberta_pretrained=False, enc=True):
        super(CodeBERTa, self).__init__()

        if roberta_pretrained:
            self.encoder = RobertaModel.from_pretrained(
                "huggingface/CodeBERTa-small-v1")
            decoder_config = OpenAIGPTConfig(vocab_size=52000)
            self.decoder = OpenAIGPTLMHeadModel(decoder_config)
        elif enc:
            num_hidden_layers = int(
                np.ceil(np.log(hidden_size + 0.) / np.log(3 + 0.)))
            intermediate_size = hidden_size
            num_attention_heads = 1

            encoder_config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size, max_position_embeddings=max_position_embeddings)


            # encoder = RobertaModel(encoder_config)
            decoder_config = GPT2Config(
                vocab_size=vocab_size, n_ctx=context_size, n_positions=context_size,
                n_embd=hidden_size, n_head=1, n_layer=2)

            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            # decoder = GPT2LMHeadModel(decoder_config)

            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
            self.model = EncoderDecoderModel(config=config)

            self.model.config.decoder.is_decoder = True
            self.model.config.decoder.add_cross_attention = True

        else:
            num_hidden_layers = int(
                np.ceil(np.log(hidden_size + 0.) / np.log(3 + 0.)))
            intermediate_size = hidden_size
            num_attention_heads = 1

            encoder_config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size, max_position_embeddings=max_position_embeddings)
            self.encoder = RobertaModel(encoder_config)

            decoder_config = GPT2Config(
                vocab_size=vocab_size, n_ctx=16, n_positions=16,
                n_embd=hidden_size, n_head=1, n_layer=2)

            self.decoder = GPT2LMHeadModel(decoder_config)

            print('ENC', '%.1E' % self.encoder.num_parameters())
            print('DEC', '%.1E' % self.decoder.num_parameters())
            print('total', '%.1E' %
                (self.encoder.num_parameters() + self.decoder.num_parameters()))

    def forward(self, x, l, decoder_attention_mask=None):
        # embed = self.encoder(x).last_hidden_state
        # out = self.decoder(inputs_embeds=embed)
        if decoder_attention_mask is None:
            return self.model(input_ids=x, decoder_input_ids=l, labels=l)
        else:
            return self.model(input_ids=x, decoder_input_ids=l*decoder_attention_mask, labels=l, decoder_attention_mask=decoder_attention_mask)

    @staticmethod
    def loss_fn(outputs, targets, batch):
        outputs = outputs.permute(0, 2, 1)
        targets = targets.permute(0, 1)
        # if outputs.shape[0] != targets.shape[0] or outputs.shape[0] * targets.shape[0] == 0:
        #     return None

        loss = F.cross_entropy(outputs, targets, reduction="none")
        mask = targets != 1
        loss = loss * mask
        return loss.sum() / batch
