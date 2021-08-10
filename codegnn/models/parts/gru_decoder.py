from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from models.parts import LuongAttention
from utils.common import PAD, SOS
from utils.vocabulary import Vocabulary


class GRUDecoder(nn.Module):
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

        self._dropout_rnn = nn.Dropout(config.rnn_dropout)
        self._sc_attn = LuongAttention(config.hidden_size)
        self._ast_attn = LuongAttention(config.hidden_size)
        self._concat_layer = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        self._norm = nn.LayerNorm(config.hidden_size)
        self._projection_layer = nn.Linear(config.hidden_size, self._out_size, bias=False)

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
        output = sc_enc.new_zeros((output_length, batch_size, self._out_size))
        #[batch_size]
        current_input = sc_enc.new_full((batch_size,), self._sos_token, dtype=torch.long)
        h_prev = initial_state
        for step in range(output_length):
            current_output, h_prev = self.decoder_step(current_input, h_prev, (sc_enc, ast_enc))
            output[step] = current_output
            if target_sequence is not None and torch.rand(1) < self._teacher_forcing:
                current_input = target_sequence[step]
            else:
                current_input = current_output.argmax(dim=-1)

        return output

    def decoder_step(
            self,
            input_tokens: torch.Tensor,  # [batch_size]
            hidden_state: torch.Tensor,  # [n_layer, batch_size, hidden_size]
            batched_context: Tuple[torch.Tensor, torch.Tensor]  # [batch_size, source_len, hidden_size], [batch_size, n_nodes, hidden_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sc_enc, ast_enc = batched_context
        batch_size = sc_enc.size(0)

        embedded = self._target_embedding(input_tokens).unsqueeze(1)

        rnn_output, h_prev = self._decoder_gru(embedded, hidden_state)
        # [batch_size, 1, hidden_size]
        rnn_output = self._dropout_rnn(rnn_output)

        # [batch_size, source_len]
        sc_attn_weights = self._sc_attn(h_prev[-1], sc_enc)
        # sc_attn_weights = F.softmax(torch.bmm(sc_enc, h_prev[-1].view(batch_size, -1, 1)).squeeze(-1), dim=-1)

        # [batch_size, 1, hidden_size]
        sc_context = torch.bmm(sc_attn_weights.unsqueeze(1), sc_enc)

        # ast_attn_weights = F.softmax(torch.bmm(ast_enc, h_prev[-1].view(batch_size, -1, 1)).squeeze(-1), dim=-1)
        ast_attn_weights = self._ast_attn(h_prev[-1], ast_enc)
        ast_context = torch.bmm(ast_attn_weights.unsqueeze(1), ast_enc)

        context = torch.cat((sc_context, rnn_output, ast_context), dim=2).squeeze(1)

        concat = self._concat_layer(context)
        concat = self._norm(concat)
        concat = torch.tanh(concat)

        # [batch size; vocab size]
        output = self._projection_layer(concat)

        return output, h_prev
