import os
import pickle
from tqdm import tqdm, trange

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from codexglue_dataset import CodeXGLUEDataset
from codeberta import CodeBERTa


in_len = 40
out_len = 10
cuda = True

tokenizer_input = ByteLevelBPETokenizer("vocab.json", "merges.txt",)
tokenizer_input._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer_input.token_to_id("</s>")), ("<s>", tokenizer_input.token_to_id("<s>")),
)
tokenizer_input.enable_truncation(max_length=in_len)

tokenizer_output = ByteLevelBPETokenizer("vocab.json", "merges.txt",)
tokenizer_output._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer_output.token_to_id("</s>")), ("<s>", tokenizer_output.token_to_id("<s>")),
)
tokenizer_output.enable_truncation(max_length=out_len)


if os.path.isfile(f'train_dataset_{in_len}_{out_len}.pickle'):
    with open(f'train_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        train_dataset = pickle.load(f)

    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        eval_dataset = pickle.load(f)
else:
    train_dataset = CodeXGLUEDataset(tokenizer_input, tokenizer_output, split="train", mode="docstring")
    eval_dataset = CodeXGLUEDataset(tokenizer_input, tokenizer_output, split="test", mode="docstring")

    with open(f'train_dataset_{in_len}_{out_len}.pickle', 'wb') as f:
          pickle.dump(train_dataset, f)

    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'wb') as f:
          pickle.dump(eval_dataset, f)


model = CodeBERTa()
if cuda:
    model.to("cuda")
model.train()

batch = 80

def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1]) for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels

train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate)


train_iterator = trange(0, 4, desc="Epoch")
optimizer = torch.optim.Adam(model.parameters())
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
