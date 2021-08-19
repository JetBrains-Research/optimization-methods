import json
from json import JSONDecodeError
from os.path import exists
from typing import Optional, Tuple, List, Dict

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.common import LABEL, AST, SOURCE, CHILDREN, TOKEN, PAD, NODE, SEPARATOR, UNK, SOS, EOS
from utils.vocabulary import Vocabulary

class JsonlSourceASTDataset(Dataset):
    _log_file = "bad_samples.log"

    def __init__(self, data_file: str, vocabulary: Vocabulary, config: DictConfig):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._vocab = vocabulary
        self._config = config

        self._token_unk = vocabulary.token_to_id[UNK]
        self._node_unk = vocabulary.node_to_id[UNK]
        self._label_unk = vocabulary.label_to_id[UNK]

        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_file, 'r') as file:
            for line in file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(file.encoding))
        self._n_samples = len(self._line_offsets)

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_file, 'r') as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line


    def _get_graph_features(self, ast):
        max_token_parts = self._config.max_node_token_parts
        max_nodes = self._config.max_ast_nodes
        nodes = torch.full((max_nodes,), self._vocab.node_to_id[PAD])
        node_tokens = torch.full((max_nodes, max_token_parts), self._vocab.token_to_id[PAD])
        adj_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.int8)
        for idx, node in enumerate(ast):
            if idx == max_nodes:
                break
            nodes[idx] = self._vocab.node_to_id.get(node[NODE], self._node_unk)
            sub_values = node[TOKEN].split(SEPARATOR)[: max_token_parts]
            sub_values_ids = [self._vocab.token_to_id.get(sv, self._token_unk) for sv in sub_values]
            node_tokens[idx, : len(sub_values_ids)] = torch.tensor(sub_values_ids)
            for child in node[CHILDREN]:
                if child < max_nodes:
                    adj_matrix[idx, child] = 1
                    adj_matrix[child, idx] = 1

        return (nodes, node_tokens), adj_matrix

    def _get_source_code_features(self, code):
        max_tokens = self._config.max_source_parts
        sc_tokens = torch.full((max_tokens,), self._vocab.token_to_id[PAD])
        sub_tokens = code.split(SEPARATOR)[:max_tokens]
        sc_tokens[:len(sub_tokens)] = torch.tensor(
            [self._vocab.token_to_id.get(st, self._token_unk) for st in sub_tokens]
        )
        return sc_tokens

    def _get_label(self, str_label: str) -> torch.Tensor:
        label = torch.full((self._config.max_label_parts + 1, 1), self._vocab.label_to_id[PAD])
        label[0, 0] = self._vocab.label_to_id[SOS]
        sublabels = str_label.split(SEPARATOR)[: self._config.max_label_parts]
        label[1 : len(sublabels) + 1, 0] = torch.tensor(
            [self._vocab.label_to_id.get(sl, self._label_unk) for sl in sublabels]
        )
        if len(sublabels) < self._config.max_label_parts:
            label[len(sublabels) + 1, 0] = self._vocab.label_to_id[EOS]
        return label

    def _read_sample(self, index: int) -> Optional[Dict]:
        raw_sample = self._read_line(index)
        try:
            sample = json.loads(raw_sample)
        except JSONDecodeError as e:
            with open(self._log_file, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return None
        if sample[LABEL] == "":
            with open(self._log_file, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return None
        return sample

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        sample = self._read_sample(index)
        if sample is None:
            return None

        label = self._get_label(sample[LABEL])
        (nodes, node_tokens), adj_matrix = self._get_graph_features(sample[AST])
        sc_tokens = self._get_source_code_features(sample[SOURCE])

        return sc_tokens, nodes, node_tokens, adj_matrix, label