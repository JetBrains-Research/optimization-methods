#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import random
import os
import pickle
from tqdm import tqdm, trange
import argparse
import itertools

import wandb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch_optimizer as optim

from tokenizers import Tokenizer
from transformers import get_linear_schedule_with_warmup

from codexglue_dataset import CodeXGLUEDocstringDataset
from javamed_dataset import JavaMedMethodNameDataset
from codeberta import CodeBERTa


torch.manual_seed(7)
random.seed(7)
np.random.seed(7)


in_len = 160


out_len = 16  # for codexglue
# out_len = 7  # for java-med

vocab_size = 1000
log_wandb = True
cuda = True

lang = "python"
# lang = "java-med"

epochs = 5
tokenizer_name = f"tokenizer_{lang}_{vocab_size}.json"
dataset_postfix = f'dataset_{lang}_in={in_len}_out={out_len}_vocab={vocab_size}.pickle'


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


model = CodeBERTa(hidden_size=100, context_size=in_len,
                  max_position_embeddings=512, vocab_size=vocab_size)
if cuda:
    model.to("cuda")
model.train()

batch = 64


def collate(examples):
    # data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
    #                     for x in examples], batch_first=True, padding_value=1)
    # input_ids = data[:batch]
    # labels = data[batch:]

    input_ids = pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = pad_sequence([torch.tensor(x[1]) for x in examples], batch_first=True, padding_value=1)

    return input_ids, labels


train_dataloader = DataLoader(
    train_dataset, batch_size=batch, shuffle=True, collate_fn=collate)

if log_wandb:
    wandb.init(project=f'CodeBERTa-tests', entity='dmivilensky')

parser = argparse.ArgumentParser(description='Train CodeBERTa.')
parser.add_argument('optimizer', type=str,
                    help='Method to use for optimization.')
args = parser.parse_args()

lr = 0.004
decay_gamma = 0.95
warmup_delay = 0

train_iterator = trange(0, epochs, desc="Epoch")

if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr)
elif args.optimizer == "LaSGD":
    sgd = torch.optim.SGD(model.parameters(), lr)
    optimizer = optim.Lookahead(sgd, k=5, alpha=0.5)
    optimizer.defaults = []
elif args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr)
elif args.optimizer == "LaAdam":
    adam = torch.optim.Adam(model.parameters(), lr)
    optimizer = optim.Lookahead(adam, k=5, alpha=0.5)
    optimizer.defaults = []
elif args.optimizer == "Lamb":
    input_ids, labels = next(iter(train_dataloader))
    model.loss_fn(model(input_ids.to("cuda")).logits,
                  labels.to("cuda"), batch).backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1. / 2)
    print('grad_norm', grad_norm)
    optimizer = optim.Lamb(model.parameters(), grad_norm*lr,
                           betas=(0.9, 0.999), eps=1e-8)
elif args.optimizer == "LaLamb":
    input_ids, labels = next(iter(train_dataloader))
    model.loss_fn(model(input_ids.to("cuda")).logits,
                  labels.to("cuda"), batch).backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1. / 2)
    print('grad_norm', grad_norm)
    lamb = optim.Lamb(model.parameters(), grad_norm*lr,
                      betas=(0.9, 0.999), eps=1e-8)
    optimizer = optim.Lookahead(lamb, k=5, alpha=0.5)
    optimizer.defaults = []
elif args.optimizer == "RAdam":
    optimizer = optim.RAdam(model.parameters(), lr,
                            betas=(0.9, 0.999), eps=1e-8)
elif args.optimizer == "LaRAdam":
    radam = optim.RAdam(model.parameters(), lr,
                        betas=(0.9, 0.999), eps=1e-8)
    optimizer = optim.Lookahead(radam, k=5, alpha=0.5)
    optimizer.defaults = []
else:
    raise ValueError(f"Unknown optimizer name: {args.optimizer}")

# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer, lr_lambda=lambda epoch: decay_gamma ** epoch)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=492*epochs)
iteration = 0

accum = 512 / batch
print(accum)

model.zero_grad()
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, (input_ids, labels) in enumerate(epoch_iterator):

        if True:
            outputs = None

            for i in range(out_len):
                if cuda:
                    decoder_attention_mask = torch.where(torch.range(0, out_len-1) < i, torch.ones_like(labels), torch.zeros_like(labels))
                    decoder_attention_mask = decoder_attention_mask.to("cuda")
                    
                    fw = model(input_ids.to("cuda"), labels.to("cuda"), decoder_attention_mask)

                    if outputs is None:
                        outputs = fw.logits[:, 0].unsqueeze(0)
                    else:
                        outputs = torch.cat([outputs, fw.logits[:, i].unsqueeze(0)], dim=0)
                # if cuda:
                #     fw = model(input_ids.to("cuda"), labels.to("cuda"))
                #     outputs = fw.logits
                #     loss = model.loss_fn(outputs, labels.to("cuda"), batch)
                # else:
                #     outputs = model(input_ids, labels).logits
                #     loss = model.loss_fn(outputs, labels, batch)
        # except:
        #     continue
    
        outputs = outputs.permute(1, 0, 2)
        loss = model.loss_fn(outputs, labels.to("cuda"), batch)
        loss = loss / accum

        if step % 20 == 0:
            for i in range(4):
                out_correct = list(itertools.takewhile(lambda x: x != 3, outputs.argmax(2)[i].tolist()))
                print('predict:', ''.join(tokenizer.decode(out_correct).split(" "))[1:].replace('\u0120', ' '))
                print(' target:', ''.join(tokenizer.decode(
                    labels[i].tolist()).split(" "))[1:].replace('\u0120', ' '))
                print()

        if loss is None:
            continue
        loss.backward()

        if (step + 1) % accum == 0:
            optimizer.step()
            model.zero_grad()

            if log_wandb and iteration % 5 == 0:
                wandb.log({"loss": loss})

            if warmup_delay != 0 and args.optimizer in ["Lamb", "LaLamb"] and iteration == warmup_delay:
                for group in optimizer.param_groups:
                    group['lr'] = lr

            iteration += 1

            scheduler.step()

    os.makedirs(f"./models/codexglue-{lang}/" +
                args.optimizer, exist_ok=True)
    with open(
        f'./models/codexglue-{lang}/' + args.optimizer +
        '/checkpoint_' + str(iteration) + '.pickle', 'wb'
    ) as f:
        pickle.dump(model, f)

    print("=== validation ===")
    model.eval()

    eval_loss = 0.0
    eval_steps = 0
    preds = np.empty((0), dtype=np.int64)
    out_label_ids = np.empty((0), dtype=np.int64)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch, collate_fn=collate)
    for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
        with torch.no_grad():
            try:
                outputs = None

                for i in range(out_len):
                    if cuda:
                        decoder_attention_mask = torch.where(torch.range(0, out_len-1) < i, torch.ones_like(labels), torch.zeros_like(labels))
                        decoder_attention_mask = decoder_attention_mask.to("cuda")
                        fw = model(input_ids.to("cuda"), labels.to("cuda"), decoder_attention_mask)

                        if outputs is None:
                            outputs = fw.logits[:, 0].unsqueeze(0)
                        else:
                            outputs = torch.cat([outputs, fw.logits[:, i].unsqueeze(0)], dim=0)
            except:
                continue
            outputs = outputs.permute(1, 0, 2)
            loss = model.loss_fn(outputs, labels.to("cuda"), batch)

            if step == 0:
                for i in range(4):
                    out_correct = list(itertools.takewhile(lambda x: x != 3, outputs.argmax(2)[i].tolist()))[:out_len]
                    print('predict:', ''.join(tokenizer.decode(out_correct).split(" "))[1:].replace('\u0120', ' '))
                    print(' target:', ''.join(tokenizer.decode(
                        labels[i].tolist()).split(" "))[1:].replace('\u0120', ' '))
                    print()

            if loss is None:
                continue

            eval_loss += loss.mean().item()
            eval_steps += 1

    eval_loss = eval_loss / eval_steps
    print("=== validation: loss ===", eval_loss)
    if log_wandb:
        wandb.log({"val/loss": eval_loss})
    model.train()
