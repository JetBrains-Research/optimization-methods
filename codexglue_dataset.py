from pathlib import Path
import os
import pandas as pd
from torch.utils.data.dataset import Dataset


class CodeXGLUEDataset(Dataset):
    def __init__(
        self, 
        tokenizer_input, tokenizer_output,
        split: str = "train", 
        mode: str = "lang-id", 
        langs: list = ["java", "python"]
    ):
        self.examples = []

        src_files = []
        for language in langs:
            src_files += list(
                Path(os.getcwd() + "/dataset/").glob(f"{language}/{split}.jsonl")
            )
        
        for src_file in src_files:
            df = pd.read_json(src_file, orient='records', lines=True)
                
            if mode == "lang-id":
                label_idx = langs.index(src_file.parents[0].name)
                
                self.examples += list(zip(
                    map(lambda x: x.ids,
                        tokenizer_input.encode_batch(df["code"].tolist())),
                    itertools.repeat(label_idx)
                ))
                
            elif mode == "docstring":
                self.examples += list(zip(
                    map(lambda x: x.ids,
                        tokenizer_input.encode_batch(df["code"].tolist())),
                    map(lambda x: x.ids,
                        tokenizer_output.encode_batch(
                            list(map(
                                lambda x: " ".join(x), 
                                df["docstring_tokens"].tolist()
                            ))
                        ))
                ))
                
            print("Ready", src_file)
        
        print("Ready all.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
