# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from torch.nn import Module
from transformers import RobertaModel, OpenAIGPTConfig, OpenAIGPTLMHeadModel, RobertaConfig
import torch.nn.functional as F
import numpy as np


class CodeBERTa(Module):
    def __init__(self, hidden_size=192, context_size=80, max_position_embeddings=256, vocab_size=10_000, roberta_pretrained=False):
        super(CodeBERTa, self).__init__()

        if roberta_pretrained:
            self.encoder = RobertaModel.from_pretrained(
                "huggingface/CodeBERTa-small-v1")
            decoder_config = OpenAIGPTConfig(vocab_size=52000)
            self.decoder = OpenAIGPTLMHeadModel(decoder_config)
        else:
            num_hidden_layers = int(
                np.ceil(np.log(hidden_size + 0.) / np.log(3 + 0.)))
            intermediate_size = 4 * hidden_size
            num_attention_heads = hidden_size // 64

            encoder_config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size, max_position_embeddings=max_position_embeddings)
            self.encoder = RobertaModel(encoder_config)
            decoder_config = OpenAIGPTConfig(
                vocab_size=vocab_size, n_ctx=context_size, n_positions=context_size,
                n_embd=hidden_size, n_head=2, n_layer=2)
            self.decoder = OpenAIGPTLMHeadModel(decoder_config)

        print('ENC', '%.1E' % self.encoder.num_parameters())
        print('DEC', '%.1E' % self.decoder.num_parameters())
        print('total', '%.1E' %
              (self.encoder.num_parameters() + self.decoder.num_parameters()))

    def forward(self, x):
        embed = self.encoder(x).last_hidden_state
        out = self.decoder(inputs_embeds=embed)
        return out

    @staticmethod
    def loss_fn(outputs, targets, batch):
        outputs = outputs.permute(0, 2, 1)
        if outputs.shape[0] != targets.shape[0] or outputs.shape[0] * targets.shape[0] == 0:
            return None

        loss = F.cross_entropy(outputs, targets, reduction="none")
        mask = targets != 1
        loss = loss * mask
        return loss.sum() / batch
