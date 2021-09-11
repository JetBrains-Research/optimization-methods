#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import BertProcessing
import argparse

from pathlib import Path
import os
import ast
import pandas as pd
import itertools

parser = argparse.ArgumentParser(description='Train tokenizer.')
parser.add_argument('lang', type=str,
                    help='Language (python, java, etc.)')
parser.add_argument('vocab_size', type=int,
                    help='Size of vocabulary for tokenizer.')
args = parser.parse_args()

texts_name = f"aggregated_texts_code_wordlevel_{args.lang}.txt"
name = f"tokenizer_wordlevel_{args.lang}_{args.vocab_size}.json"

if not (os.path.isfile(texts_name) or os.path.isfile(name)):
    print('Process dataset...')

    file_content = []

    src_files = []
    for split in ["train", "test", "val"]:
        for language in [args.lang]:
            src_files += list(
                Path(os.getcwd() +
                     "/dataset/").glob(f"{language}/{split}.jsonl")
            )

    for src_file in src_files:
        if args.lang == 'java-med':
            with open(src_file) as f:
                lines = list(map(lambda x: dict(ast.literal_eval(x)), f.readlines()))
            df = {
                "code_tokens": [line["code_tokens"] for line in lines],
                "method_name_tokens": [line["method_name_tokens"] for line in lines]
            }
        else:
            df = pd.read_json(src_file, orient='records', lines=True)

        if args.lang == 'java-med':
            file_content += list(map(
                lambda x: " ".join(x),
                df["code_tokens"]
            ))
            file_content += list(map(
                lambda x: " ".join(x),
                df["method_name_tokens"]
            ))
        else:
            file_content += df["code"].tolist()
        print(src_file, 'ready')

    textfile = open(texts_name, "w")
    for element in file_content:
        textfile.write(element + "\n")
    textfile.close()

    print('Text database generated.')

if not os.path.isfile(name):
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    trainer = WordPieceTrainer(
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"], vocab_size=args.vocab_size)

    files = [texts_name]
    tokenizer.train(files, trainer)

    tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")
         ), ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.save(name)
else:
    tokenizer = Tokenizer.from_file(name)

print('Tokenizer is trained, vocab_size:')
print(tokenizer.get_vocab_size())
texxt = u'Return either the full or truncated version'
print(f'Example of tokenization for the phrase \'{texxt}\':')
ids = tokenizer.encode(texxt).ids
print(ids)
print(tokenizer.decode(ids))
# print(''.join(tokenizer.decode(ids).split(" "))[1:].replace('\u0120', ' '))
