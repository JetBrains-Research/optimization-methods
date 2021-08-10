from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.parts import TimeDistributed
from utils.common import PAD, SOS
from utils.vocabulary import Vocabulary


class LeClairGRUDecoder(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._out_size = len(vocabulary.label_to_id)
        self._sos_token = vocabulary.label_to_id[SOS]
        self._decoder_num_layers = config.decoder_num_layers
        self._teacher_forcing = config.teacher_forcing

        self._target_embedding = nn.Embedding(
            len(vocabulary.label_to_id), config.embedding_size, padding_idx=vocabulary.label_to_id[PAD]
        )

        self._decoder_gru = nn.GRU(
            config.embedding_size,
            config.hidden_size,
            config.decoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )

        self._tdd = TimeDistributed(nn.Linear(3 * config.hidden_size, 256))  # tdddim = 256
        self._linear = nn.Linear((config.max_label_parts + 1) * 256, len(vocabulary.label_to_id))

    def forward(
            self,
            sc_enc: torch.Tensor,
            ast_enc: torch.Tensor,
            initial_state: torch.Tensor,
            output_length: int,
            target_sequence: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = sc_enc.size(0)
        # [output_length; batch_size, vocab_size]
        output_ = sc_enc.new_zeros((output_length, batch_size, self._out_size))
        input_ = sc_enc.new_zeros((output_length, batch_size,), dtype=torch.long)
        input_[0] = sc_enc.new_full((batch_size,), self._sos_token)
        for step in range(output_length):
            current_output = self.decoder_step(input_.clone(), initial_state, (sc_enc, ast_enc))
            output_[step] = current_output
            if step < output_length - 1:
                if target_sequence is not None and torch.rand(1) < self._teacher_forcing:
                    input_[step + 1] = target_sequence[step + 1]
                else:
                    input_[step + 1] = current_output.argmax(dim=-1)

        return output_

    def decoder_step(
            self,
            input_tokens: torch.Tensor,  # [batch_size]
            hidden_state: torch.Tensor,  # [n_layer, batch_size, hidden_size]
            batched_context: Tuple[torch.Tensor, torch.Tensor]  # [batch_size, source_len, hidden_size], [batch_size, n_nodes, hidden_size]
    ) -> torch.Tensor:
        sc_enc, ast_enc = batched_context

        batch_size = sc_enc.size(0)

        target_emb = self._target_embedding(input_tokens.transpose(0, 1))

        # print(f"h0: {sc_h.size()}, target emb: {target_emb.size()}")

        dec_out, _ = self._decoder_gru(target_emb, hidden_state)

        sc_attn = F.softmax(torch.bmm(dec_out, sc_enc.transpose(1, 2)), dim=-1)
        sc_context = torch.bmm(sc_attn, sc_enc)

        ast_attn = F.softmax(torch.bmm(dec_out, ast_enc.transpose(1, 2)), dim=-1)
        ast_context = torch.bmm(ast_attn, ast_enc)

        context = torch.cat((sc_context, dec_out, ast_context), dim=-1)

        out = F.relu(self._tdd(context))

        return self._linear(out.view(batch_size, -1))
