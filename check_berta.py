import os
import pickle
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


in_len = 80
out_len = 10

iteration = 7869
model = None

with open('./models/CodeBERTa-docstrings/' + 'checkpoint_' + str(iteration) + '.pickle', 'rb') as f:
    model = pickle.load(f)

batch = 32


def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
                        for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels


if os.path.isfile(f'eval_dataset_{in_len}_{out_len}.pickle'):
    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        eval_dataset = pickle.load(f)

eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch, collate_fn=collate)

for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
    with torch.no_grad():
        outputs = model(input_ids).logits
        loss = model.loss_fn(outputs, labels, batch)

        print('predict:', outputs.argmax(2))
        print(' target:', labels)
        print()

        if step > 5:
            break
