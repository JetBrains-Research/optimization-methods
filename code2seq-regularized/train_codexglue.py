from os.path import join

import hydra
import torch
from code2seq.dataset import PathContextDataModule
from code2seq_standard import Code2Seq
from code2seq.utils.vocabulary import Vocabulary
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="configs", config_name="codexglue-docstrings-py")
def train(config: DictConfig):
    vocabulary_path = join(
        config.data_folder, config.dataset.name, config.vocabulary_name)
    vocabulary = Vocabulary.load_vocabulary(vocabulary_path)

    data_module = PathContextDataModule(config, vocabulary)
    model = Code2Seq(config, vocabulary, next(iter(data_module.train_dataloader())))

    wandb_logger = WandbLogger(
        project=f"code2seq-{config.name}-final", log_model=True, offline=config.log_offline
    )
    wandb_logger.watch(model, log_freq=5)

    gpu = [0] if torch.cuda.is_available() else None
    trainer = Trainer(max_epochs=config.hyper_parameters.n_epochs,
                      logger=wandb_logger, gpus=gpu)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
