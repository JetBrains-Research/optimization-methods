from os import path
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_module.jsonl_dataset import JsonlSourceASTDataset
from utils.common import download_dataset
from utils.vocabulary import Vocabulary


class JsonlDataModule(LightningDataModule):
    _vocabulary_file = "vocabulary.pkl"
    _train = "train"
    _val = "val"
    _test = "test"

    _known_datasets = {
        "codexglue-docstrings-java": "https://www.dropbox.com/s/ohiofvwv5654m1v/codexglue-docstrings-java.tar.gz?dl=1",
        "codexglue-docstrings-py": "https://www.dropbox.com/s/nzejvu3xuy2ds23/codexglue-docstrings-py.tar.gz?dl=1",
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        self._config = config
        self._dataset_dir = path.join(config.data_folder, config.dataset)
        self._vocabulary: Optional[Vocabulary] = None

    def prepare_data(self):
        if path.exists(self._dataset_dir):
            print("Dataset is already downloaded")
            return
        print(f"Couldn't find dataset {self._config.dataset} in {self._config.data_folder}. Trying to download it.")
        if self._config.dataset not in self._known_datasets:
            print(f"Unknown dataset name: {self._config.dataset}.\n"
                  f"Try one of the following: {', '.join(self._known_datasets.keys())}")
            return
        download_dataset(
            self._known_datasets[self._config.dataset],
            self._dataset_dir,
            self._config.dataset
        )

    def setup(self, stage: Optional[str] = None):
        if not path.exists(path.join(self._dataset_dir, Vocabulary.vocab_file)):
            Vocabulary.build_from_scratch(path.join(self._dataset_dir, f"{self._config.dataset}.{self._train}.jsonl"))
        self._vocabulary = Vocabulary(self._config)

    @staticmethod
    def _collate_batch(
            sample_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sc_tokens, nodes, node_tokens, edges, labels = zip(*filter(lambda sample: sample is not None, sample_list))
        return (
                    torch.stack(sc_tokens),
                    torch.stack(nodes),
                    torch.stack(node_tokens),
                    torch.stack(edges),
                    torch.cat(labels, dim=1)
        )

    def _shared_dataloader(self, holdout: str, shuffle: bool) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating dataloaders")
        holdout_file = path.join(self._dataset_dir, f"{self._config.dataset}.{holdout}.jsonl")
        dataset = JsonlSourceASTDataset(holdout_file, self._vocabulary, self._config)
        return DataLoader(
            dataset,
            self._config.hyper_parameters.batch_size,
            shuffle=shuffle,
            num_workers=self._config.num_workers,
            collate_fn=self._collate_batch,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._train, True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._val, False)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._test, False)

    def transfer_batch_to_device(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
            batch[3].to(device),
            batch[4].to(device)
        )

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise RuntimeError("Setup datamodule for initializing vocabulary")
        return self._vocabulary
