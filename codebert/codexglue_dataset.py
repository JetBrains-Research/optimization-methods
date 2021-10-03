# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from pathlib import Path
import os
import pandas as pd
import itertools
from torch.utils.data.dataset import Dataset


class CodeXGLUEDocstringDataset(Dataset):
    def __init__(
        self,
        tokenizer_input, tokenizer_output,
        split: str = "train",
        mode: str = "lang-id",
        langs: list = ["python"]
    ):
        self.examples = []

        src_files = []
        for language in langs:
            src_files += list(
                Path(os.getcwd() +
                     "/dataset/").glob(f"{language}/{split}.jsonl")
            )

        for src_file in src_files:
            df = pd.read_json(src_file, orient='records', lines=True)

            code = list(map(lambda x: x.ids,
                            tokenizer_input.encode_batch(
                                list(map(
                                    lambda x: " ".join(x),
                                    df["code_tokens"].tolist()
                                ))
                            )))

            docstring = list(map(lambda x: x.ids,
                                 tokenizer_output.encode_batch(
                                     list(map(
                                         lambda x: " ".join(itertools.takewhile(
                                             lambda token: token != ".", x)),
                                         df["docstring_tokens"].tolist()
                                     ))
                                 )))

            self.examples += list(zip(code, docstring))

            print("Ready", src_file)

        print("Ready all.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
