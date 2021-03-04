from os.path import split, join
from typing import Dict, List

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger

from utils.common import print_table


class UploadCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            experiment = logger.experiment
            root_dir, _ = split(self._checkpoint_dir)
            experiment.save(join(self._checkpoint_dir, "*.ckpt"), base_path=root_dir)


class PrintEpochResultCallback(Callback):
    def __init__(self, *groups: str):
        self._groups = groups

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        metrics_to_print: Dict[str, List[str]] = {group: [] for group in self._groups}
        for key, value in trainer.callback_metrics.items():
            if "/" not in key:
                continue
            group, metric = key.split("/")
            if group in metrics_to_print:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics_to_print[group].append(f"{metric}={round(value, 2)}")
        print_table(metrics_to_print)
