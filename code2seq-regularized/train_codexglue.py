from os.path import join

import hydra
import torch
from code2seq.dataset import PathContextDataModule
from code2seq.model import Code2Seq
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="configs", config_name="codexglue")
def train(config: DictConfig):
    vocabulary_path = join(
        config.data_folder, config.dataset.name, config.vocabulary_name)
    vocabulary = Vocabulary.load_vocabulary(vocabulary_path)

    model = Code2Seq(config, vocabulary)
    data_module = PathContextDataModule(config, vocabulary)

    wandb_logger = WandbLogger(
        project=f"code2seq-{config.name}", log_model=True, offline=config.log_offline
    )
    wandb_logger.watch(model)

    gpu = [0] if torch.cuda.is_available() else None
    trainer = Trainer(max_epochs=config.hyper_parameters.n_epochs,
                      logger=wandb_logger, gpus=gpu)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
