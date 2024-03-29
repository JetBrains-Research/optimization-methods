import dgl
import torch
from omegaconf import DictConfig
from torch import nn

from utils.common import PAD, TOKEN, NODE, TYPE
from utils.vocabulary import Vocabulary
from utils.training import init_weights_normal, init_weights_const


class NodeEmbedding(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()

        self._token_embedding = nn.Embedding(
            len(vocabulary.token_to_id), config.embedding_size, padding_idx=vocabulary.token_to_id[PAD]
        )
        self._node_embedding = nn.Embedding(
            len(vocabulary.node_to_id), config.embedding_size, padding_idx=vocabulary.node_to_id[PAD]
        )
        # self.init_weights(how=config.initialization, value=config.init_value)

    def init_weights(self, how=None, value=None):
        if how == 'const':
            assert value is not None
            init_weights_const(self._token_embedding, value)
            init_weights_const(self._node_embedding, value)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # [n nodes; embedding size]
        token_embedding = self._token_embedding(graph.ndata[TOKEN]).sum(1)
        # [n nodes; embedding size]
        node_embedding = self._node_embedding(graph.ndata[NODE])
        # [n nodes; 2 * embedding size]
        return token_embedding + node_embedding


class TypedNodeEmbedding(NodeEmbedding):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__(config, vocabulary)
        self._type_embedding = nn.Embedding(
            len(vocabulary.type_to_id), config.embedding_size, padding_idx=vocabulary.type_to_id[PAD]
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        token_emb = self._token_embedding(graph.ndata[TOKEN]).sum(1)
        type_emb = self._type_embedding(graph.ndata[TYPE]).sum(1)
        node_emb = self._node_embedding(graph.ndata[NODE])
        return token_emb + type_emb + node_emb
