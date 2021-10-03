import random
import os
import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser
import itertools

import wandb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch_optimizer as optim

from tokenizers import Tokenizer

from codexglue_dataset import CodeXGLUEDocstringDataset
from codeberta import CodeBERTa
from word_tokenizer import WordTokenizer, WordTokenizerResponse


torch.manual_seed(7)
random.seed(7)
np.random.seed(7)


vocab_size = 5000
cuda = True
lang = "java-med"

in_len = 80
# out_len = 16  # for codexglue
out_len = 7  # for java-med


tokenize_words = True
log_wandb = True

# lang = "python"
lang = "java-med"

tokenizer_name = f"tokenizer_{lang}_{vocab_size}" + ("_word" if tokenize_words else "") + (".pkl" if tokenize_words else ".json")
dataset_postfix = f'dataset_{lang}_in={in_len}_out={out_len}_vocab={vocab_size}' + ("_word" if tokenize_words else "") + '.pickle'


if tokenize_words:
    tokenizer = WordTokenizer(tokenizer_name, pretrained=True)
    vocab_size = tokenizer.get_vocab_size()

    tokenizer_input = WordTokenizer(tokenizer_name, pretrained=True)
    tokenizer_input.enable_truncation(max_length=in_len)

    tokenizer_output = WordTokenizer(tokenizer_name, pretrained=True)
    tokenizer_output.enable_truncation(max_length=out_len)
else:
    tokenizer = Tokenizer.from_file(tokenizer_name)

    tokenizer_input = Tokenizer.from_file(tokenizer_name)
    tokenizer_input.enable_truncation(max_length=in_len)

    tokenizer_output = Tokenizer.from_file(tokenizer_name)
    tokenizer_output.enable_truncation(max_length=out_len)

if os.path.isfile('train_' + dataset_postfix):
    with open('train_' + dataset_postfix, 'rb') as f:
        train_dataset = pickle.load(f)

    with open('eval_' + dataset_postfix, 'rb') as f:
        eval_dataset = pickle.load(f)
else:
    print('Process dataset...')
    if lang == "java-med":
        train_dataset = JavaMedMethodNameDataset(
            tokenizer_input, tokenizer_output, split="train")
        eval_dataset = JavaMedMethodNameDataset(
            tokenizer_input, tokenizer_output, split="val")
    else:
        train_dataset = CodeXGLUEDocstringDataset(
            tokenizer_input, tokenizer_output, langs=[lang], split="train")
        eval_dataset = CodeXGLUEDocstringDataset(
            tokenizer_input, tokenizer_output, langs=[lang], split="test")

    with open('train_' + dataset_postfix, 'wb') as f:
        pickle.dump(train_dataset, f)

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

batch = 128

def collate(examples):
    input_ids = pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = pad_sequence([torch.tensor(x[1]) for x in examples], batch_first=True, padding_value=1)

    return input_ids, labels

eval_dataloader = DataLoader(eval_dataset, batch_size=batch, collate_fn=collate)

hyps, refs = [], []

for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
    with torch.no_grad():
        outputs = None

        for i in range(min(labels.shape[1], out_len)):
            if cuda:
                decoder_attention_mask = torch.where(torch.arange(0, min(labels.shape[1], out_len)) < i, torch.ones_like(labels), torch.zeros_like(labels))
                decoder_attention_mask = decoder_attention_mask.to("cuda")
                fw = model(input_ids.to("cuda"), labels.to("cuda"), decoder_attention_mask)

                if outputs is None:
                    outputs = fw.logits[:, 0].unsqueeze(0)
                else:
                    outputs = torch.cat([outputs, fw.logits[:, i].unsqueeze(0)], dim=0)

        outputs = outputs.permute(1, 0, 2)
        
        for i in range(len(labels)):
            out_correct = list(itertools.takewhile(lambda x: x != 3, outputs.argmax(2)[i].tolist()))[:out_len]
            hyp = tokenizer.decode(out_correct)
            ref = tokenizer.decode(labels[i].tolist())
                
            if len(ref.strip()) > 0:
                if len(hyp.strip()) == 0:
                    hyp = 'xxx'
                
                hyps.append(hyp)
                refs.append(ref)

print('Ready', args.checkpoint.split("/")[-2])
dirs = "outputs"
os.makedirs("./" + dirs, exist_ok=True)
with open("./" + dirs + "/" + args.checkpoint.split("/")[-2] + "_test_outputs.pkl", 'wb') as f:
    pickle.dump((hyps, refs), f)
