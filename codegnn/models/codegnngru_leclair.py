from typing import Tuple, List, Dict, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

from models.parts import GCNLayer, LeClairGRUDecoder
from utils.common import PAD, SOS, EOS, TOKEN, NODE
from utils.training import configure_optimizers_alon
from utils.vocabulary import Vocabulary
from utils.metrics import PredictionStatistic



class LeClairCodeGNNGRU(LightningModule):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self._vocabulary = vocabulary

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        self._label_pad_id = vocabulary.label_to_id[PAD]
        self._metric_skip_tokens = [
            vocabulary.label_to_id[i] for i in [PAD, EOS, SOS] if i in vocabulary.label_to_id
        ]

        #source code embedding and node token embeddings
        self.token_embedding = nn.Embedding(
            len(vocabulary.token_to_id), config.embedding_size, padding_idx=vocabulary.token_to_id[PAD]
        )

        #node embeddings
        self.node_embedding = nn.Embedding(
            len(vocabulary.node_to_id), config.embedding_size, padding_idx=vocabulary.node_to_id[PAD]
        )

        #Encoder
        self.source_code_enc = nn.GRU(
            config.embedding_size,
            config.hidden_size,
            config.encoder_num_layers,
            dropout=config.rnn_dropout if config.encoder_num_layers > 1 else 0,
            batch_first=True,
        )

        gcn_layers = [GCNLayer(config.embedding_size, config.gcn_hidden_size)]
        gcn_layers.extend(
            [GCNLayer(config.gcn_hidden_size, config.gcn_hidden_size) for _ in range(config.num_hops - 1)]
        )
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.ast_rnn_enc = nn.GRU(
            config.gcn_hidden_size,
            config.hidden_size,
            config.encoder_num_layers,
            dropout=config.rnn_dropout if config.encoder_num_layers > 1 else 0,
            batch_first=True,
        )

        #Decoder
        self.decoder = LeClairGRUDecoder(config, vocabulary)

        # saving predictions
        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)
        self._test_outputs = []
        self._val_outputs = []
        self.val = False

        #SWA
        self.swa = (config.hyper_parameters.optimizer == "SWA")

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._config.hyper_parameters, self.parameters())

    def forward(
            self,
            source_code,
            ast_nodes,
            ast_node_tokens,
            ast_edges,
            target = None,
    ):
        batch_size = source_code.size(0)

        sc_emb = self.token_embedding(source_code)

        ast_node_emb = self.node_embedding(ast_nodes) + self.token_embedding(ast_node_tokens).sum(2)  # no second term in original implementation, but why not

        sc_enc, sc_h = self.source_code_enc(sc_emb)
        ast_enc = ast_node_emb
        for i in range(self._config.num_hops):
            ast_enc = self.gcn_layers[i](ast_enc, ast_edges)
        ast_enc, _ = self.ast_rnn_enc(ast_enc, sc_h)

        # output_logits = self.decoder(
        #     sc_enc,
        #     ast_enc,
        #     sc_h,
        #     self._config.max_label_parts + 1,
        #     target
        # )

        output_logits = self.decoder(
            sc_enc,
            ast_enc,
            sc_h,
            self._config.max_label_parts + 1,
            target
        )
        return output_logits


    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate cross entropy ignoring PAD index
        :param logits: [seq_len; batch_size; vocab_size]
        :param labels: [seq_len; batch_size]
        :return: [1]
        """
        batch_size = labels.shape[-1]
        _logits = logits.permute(1, 2, 0)
        _labels = labels.transpose(0, 1)
        loss = F.cross_entropy(_logits, _labels, reduction="none")
        mask = _labels != self._vocabulary.label_to_id[PAD]
        loss = loss * mask
        loss = loss.sum() / batch_size
        return loss

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
    ) -> Dict:
        source_code, ast_nodes, ast_node_tokens, ast_edges, labels = batch
        logits = self(source_code, ast_nodes, ast_node_tokens, ast_edges, labels)
        labels = batch[-1]
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        batch_metric = statistic.update_statistic(labels, prediction)

        log: Dict[str, Union[float, torch.Tensor]] = {'train/loss': loss}
        for key, value in batch_metric.items():
            log[f"train/{key}"] = value
        self.log_dict(log)
        self.log("f1", batch_metric["f1"], prog_bar=True, logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            test: bool = False,
    ) -> Dict:
        if self.swa and not self.val:
            print("Validation starts")
            self.trainer.optimizers[0].swap_swa_sgd()
            self.val = True
        source_code, ast_nodes, ast_node_tokens, ast_edges, labels = batch
        logits = self(source_code, ast_nodes, ast_node_tokens, ast_edges)
        labels = batch[-1]
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        if test:
            self._test_outputs.append(prediction.detach().cpu())
        else:
            self._val_outputs.append(prediction.detach().cpu())

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        statistic.update_statistic(labels, prediction)

        return {"loss": loss, "statistic": statistic}

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
    ) -> Dict:
        return self.validation_step(batch, batch_idx, test=True)

        # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            statistic = PredictionStatistic.create_from_list([out["statistic"] for out in outputs])
            epoch_metrics = statistic.get_metric()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            for key, value in epoch_metrics.items():
                log[f"{group}/{key}"] = value
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")
        torch.save(self._val_outputs,
                   f"{self._config.output_dir}/{self._config.hyper_parameters.optimizer}_epoch{self.current_epoch}_val_outputs.pkl")
        self._val_outputs = []
        print("Validation finished")
        if self.swa:
            self.trainer.optimizers[0].swap_swa_sgd()
            self.val = False

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
        torch.save(self._test_outputs,
                   f"{self._config.output_dir}/{self._config.hyper_parameters.optimizer}_test_outputs.pkl")
