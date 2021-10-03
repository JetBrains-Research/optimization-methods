# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from pathlib import Path
import os
import ast
import pandas as pd
import itertools
from tqdm import tqdm
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
            lines = []
            if split == "train":
                num_lines = int(0.40 * sum(1 for line in open(src_file)))
            else:
                num_lines = sum(1 for line in open(src_file))
            with open(src_file) as f:
                if split == "train":
                    for line in tqdm(itertools.islice(f, num_lines), total=num_lines):
                        lines.append(dict(ast.literal_eval(line)))
                else:
                    for line in tqdm(f, total=num_lines):
                        lines.append(dict(ast.literal_eval(line)))
            df = {
                "code_tokens": [line["SOURCE"].split("|") for line in lines],
                "method_name_tokens": [line["label"].split("|") for line in lines]
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
