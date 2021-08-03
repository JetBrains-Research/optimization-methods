import os
import pickle
from tqdm import tqdm, trange

import wandb
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from codexglue_dataset import CodeXGLUEDataset
from codeberta import CodeBERTa


in_len = 80
out_len = 10
cuda = True

tokenizer_back = ByteLevelBPETokenizer("vocab.json", "merges.txt",)

tokenizer_input = ByteLevelBPETokenizer("vocab.json", "merges.txt",)
tokenizer_input._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer_input.token_to_id("</s>")
     ), ("<s>", tokenizer_input.token_to_id("<s>")),
)
tokenizer_input.enable_truncation(max_length=in_len)

tokenizer_output = ByteLevelBPETokenizer("vocab.json", "merges.txt",)
tokenizer_output._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer_output.token_to_id("</s>")
     ), ("<s>", tokenizer_output.token_to_id("<s>")),
)
tokenizer_output.enable_truncation(max_length=out_len)


if os.path.isfile(f'train_dataset_{in_len}_{out_len}.pickle'):
    with open(f'train_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        train_dataset = pickle.load(f)

    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        eval_dataset = pickle.load(f)
else:
    train_dataset = CodeXGLUEDataset(
        tokenizer_input, tokenizer_output, split="train", mode="docstring")
    eval_dataset = CodeXGLUEDataset(
        tokenizer_input, tokenizer_output, split="test", mode="docstring")

    with open(f'train_dataset_{in_len}_{out_len}.pickle', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'wb') as f:
        pickle.dump(eval_dataset, f)


model = CodeBERTa()
if cuda:
    model.to("cuda")
model.train()

batch = 64


def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
                        for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels


train_dataloader = DataLoader(
    train_dataset, batch_size=batch, shuffle=True, collate_fn=collate)

wandb.init(project='CodeBERTa', entity='dmivilensky')

train_iterator = trange(0, 5, desc="Epoch")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

        if iteration % 5 == 0:
            wandb.log({"loss": loss})

        iteration += 1

    os.makedirs("./models/CodeBERTa-docstrings", exist_ok=True)
    with open('./models/CodeBERTa-docstrings/' + 'checkpoint_' + str(iteration) + '.pickle', 'wb') as f:
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
                    print('predict:', tokenizer_back.decode(
                        outputs.argmax(2)[i].tolist()))
                    print(' target:', tokenizer_back.decode(labels[i].tolist()))
                    print()

            if loss is None:
                continue

            eval_loss += loss.mean().item()
            eval_steps += 1

    eval_loss = eval_loss / eval_steps
    print("=== validation: loss ===", eval_loss)
    wandb.log({"val/loss": eval_loss})
    model.train()
