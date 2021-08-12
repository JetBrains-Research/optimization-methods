from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import BertProcessing
import argparse

from pathlib import Path
import os
import pandas as pd
import itertools

parser = argparse.ArgumentParser(description='Train tokenizer.')
parser.add_argument('vocab_size', type=int,
                    help='Size of vocabulary for tokenizer.')
args = parser.parse_args()

texts_name = "aggregated_texts_code.txt"
name = f"small_tokenizer_{args.vocab_size}_clear.json"

if not (os.path.isfile(texts_name) or os.path.isfile(name)):
    print('Process dataset...')

    file_content = []

    src_files = []
    for split in ["train", "test", "valid"]:
        for language in ["python"]:
            src_files += list(
                Path(os.getcwd() +
                     "/dataset/").glob(f"{language}/{split}.jsonl")
            )

    for src_file in src_files:
        df = pd.read_json(src_file, orient='records', lines=True)
        file_content += df["code"].tolist()
        print(src_file, 'ready')

    textfile = open(texts_name, "w")
    for element in file_content:
        textfile.write(element + "\n")
    textfile.close()

    print('Text database generated.')

if not os.path.isfile(name):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"], vocab_size=args.vocab_size)
    tokenizer.pre_tokenizer = ByteLevel()

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
print('Example of tokenization for the phrase \'Let us try with this text\':')
ids = tokenizer.encode(u"Let us Dailymotion try with this text").ids
print(ids)
print(''.join(tokenizer.decode(ids).split(" "))[1:].replace('\u0120', ' '))
