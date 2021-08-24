#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import os
import pickle
import itertools
from collections import defaultdict

import numpy as np
import torch

from metrics import Metrics
from scipy.stats import wilcoxon, mannwhitneyu
from time import perf_counter


def get_hyps_refs(predict_file="./outputs/SGD_test_outputs.pkl"):
    start = perf_counter()

    with open(predict_file, 'rb') as f:
        hyps, refs = pickle.load(f)

    finish = perf_counter()
    print(f"Preprocessing time elapsed: {finish - start}, s")

    return hyps, refs


long_perspective = True
if long_perspective:
    dirs = 'outputs_long'
else:
    dirs = 'outputs'

global_methods = [
    ('SGD', './' + dirs + '/SGD_test_outputs.pkl'),
    ('LaSGD', './' + dirs + '/LaSGD_test_outputs.pkl'),

    ('Adam', './' + dirs + '/Adam_test_outputs.pkl'),
    ('LaAdam', './' + dirs + '/LaAdam_test_outputs.pkl'),

    ('RAdam', './' + dirs + '/RAdam_test_outputs.pkl'),
    ('LaRAdam', './' + dirs + '/LaRAdam_test_outputs.pkl'),

    ('Lamb', './' + dirs + '/Lamb_test_outputs.pkl'),
    ('LaLamb', './' + dirs + '/LaLamb_test_outputs.pkl')
]

dict_filename = './' + dirs + '/global_methods.data'

if os.path.isfile(dict_filename):
    with open(dict_filename, 'rb') as fp:
        global_methods_report = pickle.load(fp)

else:
    global_methods_report = {}

    for method, path in global_methods:
        hyps, refs = get_hyps_refs(predict_file=path)
        metrics = Metrics(hyps, refs)
        global_methods_report[method] = metrics.get_statistics(
            stats={'bleu', 'rouge', 'chrF', 'chrFpp', 'prec_rec_f1'},
            verbose=True, with_symbolic=True, bert=True)
        print(method, global_methods_report[method])

    with open(dict_filename, 'wb') as fp:
        pickle.dump(global_methods_report, fp)
