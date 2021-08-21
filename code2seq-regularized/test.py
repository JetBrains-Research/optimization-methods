from argparse import ArgumentParser
from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from dataset import PathContextDataModule, TypedPathContextDataModule
from code2seq_standard import Code2Seq
from utils.vocabulary import Vocabulary


def load_code2seq(
    checkpoint_path: str, config: DictConfig, vocabulary: Vocabulary
) -> Tuple[Code2Seq, PathContextDataModule]:
    
    return model, data_module

def test(checkpoint_path: str, data_folder: str = None, batch_size: int = None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]["config"]
    vocabulary = checkpoint["hyper_parameters"]["vocabulary"]

    model = Code2Seq.load_from_checkpoint(checkpoint_path=checkpoint_path)
    data_module = PathContextDataModule(config, vocabulary)

    seed_everything(config.seed)
    gpu = [0] if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    args = arg_parser.parse_args()

    test(args.checkpoint)