import os
import pickle
import argparse
from time import perf_counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
import textmetric


def get_checkpoint_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f).cpu()
    return model


def collate(examples):
    data = pad_sequence([torch.tensor(x[0]) for x in examples] + [torch.tensor(x[1])
                        for x in examples], batch_first=True, padding_value=1)
    input_ids = data[:batch]
    labels = data[batch:]

    return input_ids, labels


def ids_to_text(string):
    res = []
    for idx in string:
        if idx in ["<pad>", "<s>"]:
            continue
        elif idx == "</s>":
            return res
        res.append(idx)
    return res


def get_hyps_refs(eval_dataloader, model, tokenizer):
    start = perf_counter()
    hyps = []
    refs = []

    for input_ids, labels in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids).logits

            for i in range(batch):
                res_our = ids_to_text(tokenizer.decode(
                    outputs.argmax(2)[i].tolist()))
                res_ref = ids_to_text(tokenizer.decode(labels[i].tolist()))

                if len(res_ref) > 0:
                    if len(res_our) == 0:
                        res_our = ['xxx']
                    hyps.append(' '.join(res_our))
                    refs.append(' '.join(res_ref))

    finish = perf_counter()
    print(f"Preprocessing time elapsed: {finish - start}, s")

    return hyps, refs


in_len = 80
out_len = 10

batch = 512

if os.path.isfile(f'eval_dataset_{in_len}_{out_len}.pickle'):
    with open(f'eval_dataset_{in_len}_{out_len}.pickle', 'rb') as f:
        eval_dataset = pickle.load(f)

tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt",)


parser = argparse.ArgumentParser(description='Train CodeBERTa.')
parser.add_argument('optimizer', type=str,
                    help='Method to use for optimization.')
args = parser.parse_args()

path = './models/CodeBERTa-docstrings/' + args.optimizer + '/'

for filename in os.listdir(path):
    model = get_checkpoint_model(path + filename)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch, collate_fn=collate)

    hyps, refs = get_hyps_refs(eval_dataloader, model, tokenizer)
    res_metrics = statmetric.Metrics(hyps, refs)
    test_metrics = res_metrics.get_statistics(with_symbolic=True, bert=True)

    os.makedirs(path + 'stats/', exist_ok=True)
    with open(path + 'stats/' + filename, 'wb') as f:
        pickle.dump(test_metrics, f)
