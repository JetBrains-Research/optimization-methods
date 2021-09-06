# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from pathlib import Path
import os
import ast
import pandas as pd
import itertools
from torch.utils.data.dataset import Dataset


class JavaMedMethodNameDataset(Dataset):
    def __init__(
        self,
        tokenizer_input, tokenizer_output,
        split: str = "train"
    ):
        self.examples = []

        src_files = list(
                Path(os.getcwd() +
                     "/dataset/").glob(f"java-med/{split}.jsonl")
            )

        for src_file in src_files:
            with open(src_file) as f:
                lines = list(map(lambda x: dict(ast.literal_eval(x)), f.readlines()))
            df = {
                "code_tokens": [line["code_tokens"] for line in lines],
                "method_name_tokens": [line["method_name_tokens"] for line in lines]
            }

            code = list(map(lambda x: x.ids,
                            tokenizer_input.encode_batch(
                                list(map(
                                    lambda x: " ".join(x),
                                    df["code_tokens"]
                                ))
                            )))

            method_name = list(map(lambda x: x.ids,
                                 tokenizer_output.encode_batch(
                                     list(map(
                                         lambda x: " ".join(itertools.takewhile(
                                             lambda token: token != ".", x)),
                                         df["method_name_tokens"]
                                     ))
                                 )))

            self.examples += list(zip(code, method_name))

            print("Ready", src_file)

        print("Ready all.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
