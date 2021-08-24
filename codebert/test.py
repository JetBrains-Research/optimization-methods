import random
import os
import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser

import wandb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch_optimizer as optim

from tokenizers import Tokenizer

from codexglue_dataset import CodeXGLUEDocstringDataset
from codeberta import CodeBERTa


torch.manual_seed(7)
random.seed(7)
np.random.seed(7)


in_len = 80
out_len = 16
vocab_size = 10000
cuda = True
lang = "java"
tokenizer_name = f"tokenizer_{lang}_{vocab_size}.json"
dataset_postfix = f'dataset_{lang}_in={in_len}_out={out_len}.pickle'

long_perspective = False
if long_perspective:
    dirs = f'outputs-codexglue-{lang}-long'
else:
    dirs = f'outputs-codexglue-{lang}'

tokenizer = Tokenizer.from_file(tokenizer_name)

if os.path.isfile('eval_' + dataset_postfix):
    with open('eval_' + dataset_postfix, 'rb') as f:
        eval_dataset = pickle.load(f)
else:
    print('Process dataset...')
    eval_dataset = CodeXGLUEDocstringDataset(
        tokenizer_input, tokenizer_output, langs=[lang], split="test")

    with open('eval_' + dataset_postfix, 'wb') as f:
        pickle.dump(eval_dataset, f)

    print('Dataset instances prepared and saved.')

arg_parser = ArgumentParser()
arg_parser.add_argument("checkpoint", type=str)
args = arg_parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    model = pickle.load(f)

if cuda:
    model.to("cuda")

model.eval()

batch = 512


def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
                        for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels


eval_dataloader = DataLoader(eval_dataset, batch_size=batch, collate_fn=collate)

hyps, refs = [], []

for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
    with torch.no_grad():
        if cuda:
            outputs = model(input_ids.to("cuda")).logits
        else:
            outputs = model(input_ids).logits

        for i in range(len(labels)):
            hyp = ''.join(tokenizer.decode(outputs.argmax(2)[i].tolist()).split(" "))[1:].replace('\u0120', ' ')
            ref = ''.join(tokenizer.decode(labels[i].tolist()).split(" "))[1:].replace('\u0120', ' ')
                
            if len(ref.strip()) > 0:
                if len(hyp.strip()) == 0:
                    hyp = 'xxx'
                
                hyps.append(hyp)
                refs.append(ref)

print('Ready', args.checkpoint.split("/")[-2])
os.makedirs("./" + dirs, exist_ok=True)
with open("./" + dirs + "/" + args.checkpoint.split("/")[-2] + "_test_outputs.pkl", 'wb') as f:
    pickle.dump((hyps, refs), f)
