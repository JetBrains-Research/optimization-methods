from typing import Tuple, List, Dict, Union

import dgl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from models.parts import NodeEmbedding, LSTMDecoder, TreeLSTM
from utils.training import configure_optimizers_alon
from utils.common import PAD, UNK, EOS, SOS
from utils.metrics import PredictionStatistic
from utils.vocabulary import Vocabulary


class TreeLSTM2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self._vocabulary = vocabulary

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        self._label_pad_id = vocabulary.label_to_id[PAD]
        self._metric_skip_tokens = [
            vocabulary.label_to_id[i] for i in [PAD, UNK, EOS, SOS] if i in vocabulary.label_to_id
        ]

        self._embedding = self._get_embedding()
        self._encoder = TreeLSTM(config)
        self._decoder = LSTMDecoder(config, vocabulary)
        self._test_outputs = []

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return NodeEmbedding(self._config, self._vocabulary)

    def forward(  # type: ignore
        self,
        batched_trees: dgl.DGLGraph,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        batched_trees.ndata["x"] = self._embedding(batched_trees)
        encoded_nodes = self._encoder(batched_trees)
        output_logits = self._decoder(encoded_nodes, batched_trees.batch_num_nodes(), output_length, target_sequence)
        return output_logits
