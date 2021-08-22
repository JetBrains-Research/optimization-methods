import os
import pickle
import itertools
from collections import defaultdict

import numpy as np
import torch

from metrics import Metrics
from code2seq.dataset import PathContextDataModule

from scipy.stats import wilcoxon, mannwhitneyu
from time import perf_counter


def ids_to_text(ids):
    res = []
    for idx in ids:
        if idx in [0, 2]:  # PAD, SOS
            continue
        elif idx == 1:  # EOS
            return res
        res.append(id_to_label[idx.item()])
    return res


def get_hyps_refs(predict_file="./outputs/SGD_test_outputs.pkl",
                  use_first=10):
    start = perf_counter()
    test_dataloader = PathContextDataModule(
        config, vocabulary).test_dataloader()
    predictions = torch.load(
        f"{predict_file}", map_location=torch.device('cpu'))

    hyps = []
    refs = []

    if use_first is None:
        x = zip(predictions, test_dataloader)
    else:
        x = itertools.islice(zip(predictions, test_dataloader), use_first)

    for batch_our, batch_ref in x:
        for batch_idx in range(batch_our.size(1)):
            res_our = ids_to_text(batch_our[:, batch_idx])
            res_ref = ids_to_text(batch_ref.labels[:, batch_idx])

            if len(res_ref) > 0:
                if len(res_our) == 0:
                    res_our = ['xxx']
                hyps.append(' '.join(res_our))
                refs.append(' '.join(res_ref))
    finish = perf_counter()
    print(f"Preprocessing time elapsed: {finish - start}, s")

    return hyps, refs


with open('./data/codexglue-docstrings-py/vocabulary.pkl', 'rb') as f:
    vocub = pickle.load(f)

id_to_label = {v: k for k, v in vocub['label_to_id'].items()}

checkpoint = torch.load(
    "./code2seq-codexglue-docstrings-py-final/SGD/checkpoints/epoch=4-step=2459.ckpt", map_location=torch.device("cpu"))
config = checkpoint["hyper_parameters"]["config"]
vocabulary = checkpoint["hyper_parameters"]["vocabulary"]

global_methods = [
    ('SGD', './outputs/SGD_test_outputs.pkl'),
    ('LaSGD', './outputs/LaSGD_test_outputs.pkl'),

    ('Adam', './outputs/Adam_test_outputs.pkl'),
    ('LaAdam', './outputs/LaAdam_test_outputs.pkl'),

    ('RAdam', './outputs/RAdam_test_outputs.pkl'),
    ('LaRAdam', './outputs/LaRAdam_test_outputs.pkl'),

    ('Lamb', './outputs/Lamb_test_outputs.pkl'),
    ('LaLamb', './outputs/LaLamb_test_outputs.pkl')
]

dict_filename = './outputs/global_methods.data'

if os.path.isfile(dict_filename):
    with open(dict_filename, 'rb') as fp:
        global_methods_report = pickle.load(fp)

else:
    global_methods_report = {}

    for method, path in global_methods:
        hyps, refs = get_hyps_refs(predict_file=path, use_first=None)
        metrics = Metrics(hyps, refs)
        global_methods_report[method] = metrics.get_statistics(
            stats={'bleu', 'rouge', 'chrF', 'chrFpp', 'prec_rec_f1'}, 
            verbose=True, with_symbolic=True, bert=True)
        print(method, global_methods_report[method])

    with open(dict_filename, 'wb') as fp:
        pickle.dump(global_methods_report, fp)
