#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import pandas as pd
from textmetric.metrics import Metrics
from scipy.stats import wilcoxon
from collections import Counter


DIR = "javaxglue"

df = None
metric = "meteor"
refs = None
hyps_database = {}

for method in [
    "RAdam", "LaRAdam", "Adam", "LaAdam", "Adamax", "LaAdamax",
    "DiffGrad", "LaDiffGrad", "Lamb", "LaLamb", "SGD", "LaSGD", 
    "Yogi", "LaYogi"
]:

    with open(DIR + "/" + method + "_test_outputs.pkl", "rb") as f:
        hyps, refs = pickle.load(f)
    
    hyps_database[method] = hyps

    metrics = Metrics(hyps, refs)
    metrics = metrics.get_statistics(
        stats={metric}, 
        verbose=False, with_symbolic=False, bert=False
    )

    if df is None:
        df = pd.DataFrame([[ref] for ref in refs], columns=['refs'])
    df[method] = metrics["scores"][metric]
    df[method] = pd.to_numeric(df[method])
    print(df.head())

print(df.drop(["refs"], axis=1).idxmax(axis=1))
print(Counter(df.drop(["refs"], axis=1).idxmax(axis=1).to_numpy().tolist()))

best_methods = df.drop(["refs"], axis=1).idxmax(axis=1).to_numpy()
best_hyps = []
for i in range(len(refs)):
    best_hyps.append(hyps_database[best_methods[i]][i])


method = "RAdam"

with open(DIR + "/" + method + "_test_outputs.pkl", "rb") as f:
    hyps, refs = pickle.load(f)

metrics_single = Metrics(hyps, refs)
metrics_single = metrics_single.get_statistics(
    stats={metric}, 
    verbose=False, with_symbolic=False, bert=False
)

metrics_best = Metrics(best_hyps, refs)
metrics_best = metrics_best.get_statistics(
    stats={metric}, 
    verbose=False, with_symbolic=False, bert=False
)

print("best", metric, ":", metrics_best['score'][metric])
print(method, metric, ":", metrics_single['score'][metric])

if metrics_best['score'][metric] > metrics_single['score'][metric]:
    pc = round(100 * (metrics_best['score'][metric] - metrics_single['score'][metric]) / metrics_single['score'][metric], 2)
    print('>', f'({pc}%)', end=' ')
    h1 = 'less'
else:
    pc = round(100 * (metrics_single['score'][metric] - metrics_best['score'][metric]) / metrics_single['score'][metric], 2)
    print('<', f'({pc}%)', end=' ')
    h1 = 'greater'
                    
print()
w, p = wilcoxon(
    metrics_best['scores'][metric],
    metrics_single['scores'][metric],
    alternative=h1
)
print('wilcoxon signed-rank: w =', round(int(w), -4), ', p =', p)
if p > 0.05:
    if h1 == "less":
        print("failed to reject H0:", "best", "<" if h1 == "greater" else ">", method, ", so")
        print("\033[93m" + "best" + (" < " if h1 == "greater" else " > ") + method + "\033[0m")
    else:
        print("failed to reject H0:", "best", "<" if h1 == "greater" else ">", method, ", so")
        print("\033[92m" + "best" + (" < " if h1 == "greater" else " > ") + method + "\033[0m")
else:
    print("reject H0:", "best", "<" if h1 == "greater" else ">", method, ", so")
    print("\033[41m" + "best" + (" > " if h1 == "greater" else " < ") + method + "\033[0m")
