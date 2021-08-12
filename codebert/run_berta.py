import random
import os
import pickle
from tqdm import tqdm, trange
import argparse

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
vocab_size = 10_000
log_wandb = True
cuda = True
tokenizer_name = f"small_tokenizer_{vocab_size}_clear.json"
dataset_postfix = f'dataset_new_{in_len}_{out_len}_clear.pickle'


tokenizer = Tokenizer.from_file(tokenizer_name)

tokenizer_input = Tokenizer.from_file(tokenizer_name)
tokenizer_input.enable_truncation(max_length=in_len)

tokenizer_output = Tokenizer.from_file(tokenizer_name)
tokenizer_output.enable_truncation(max_length=out_len)


if os.path.isfile('train_' + dataset_postfix):
    with open('train' + dataset_postfix, 'rb') as f:
        train_dataset = pickle.load(f)

    with open('eval' + dataset_postfix, 'rb') as f:
        eval_dataset = pickle.load(f)
else:
    train_dataset = CodeXGLUEDocstringDataset(
        tokenizer_input, tokenizer_output, split="train")
    eval_dataset = CodeXGLUEDocstringDataset(
        tokenizer_input, tokenizer_output, split="test")

    with open('train' + dataset_postfix, 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('eval' + dataset_postfix, 'wb') as f:
        pickle.dump(eval_dataset, f)


model = CodeBERTa(hidden_size=64, context_size=in_len,
                  max_position_embeddings=256, vocab_size=vocab_size)
if cuda:
    model.to("cuda")
model.train()

batch = 512


def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
                        for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels


train_dataloader = DataLoader(
    train_dataset, batch_size=batch, shuffle=True, collate_fn=collate)

if log_wandb:
    wandb.init(project='CodeBERTa-same', entity='dmivilensky')

parser = argparse.ArgumentParser(description='Train CodeBERTa.')
parser.add_argument('optimizer', type=str,
                    help='Method to use for optimization.')
args = parser.parse_args()

lr = 0.001
decay_gamma = 0.95
warmup_delay = 0

train_iterator = trange(0, 10, desc="Epoch")

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
    lamb = optim.Lamb(model.parameters(), lr,
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

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: decay_gamma ** epoch)
iteration = 0

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, (input_ids, labels) in enumerate(epoch_iterator):
        optimizer.zero_grad()

        if cuda:
            outputs = model(input_ids.to("cuda")).logits
            loss = model.loss_fn(outputs, labels.to("cuda"), batch)
        else:
            outputs = model(input_ids).logits
            loss = model.loss_fn(outputs, labels, batch)

        if loss is None:
            continue
        loss.backward()

        optimizer.step()

        if log_wandb and iteration % 5 == 0:
            wandb.log({"loss": loss})

        if warmup_delay != 0 and args.optimizer == "Lamb" and iteration == warmup_delay:
            for group in optimizer.param_groups:
                group['lr'] = lr

        iteration += 1

    scheduler.step()

    os.makedirs("./models/CodeBERTa-docstrings-new/" +
                args.optimizer, exist_ok=True)
    with open(
        './models/CodeBERTa-docstrings-new/' + args.optimizer +
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

            if cuda:
                outputs = model(input_ids.to("cuda")).logits
                loss = model.loss_fn(outputs, labels.to("cuda"), batch)
            else:
                outputs = model(input_ids).logits
                loss = model.loss_fn(outputs, labels, batch)

            if step == 0:
                for i in range(3):
                    print('predict:', ''.join(tokenizer.decode(outputs.argmax(
                        2)[i].tolist()).split(" "))[1:].replace('\u0120', ' '))
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
