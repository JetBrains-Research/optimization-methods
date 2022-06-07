#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
from textmetric.metrics import Metrics
from scipy.stats import wilcoxon


DIR = "javaxglue"

for metric in ["chrFpp", "meteor"]:
    print("###", metric, "###")
    method_2 = "RAdam"

    with open(DIR + "/" + method_2 + "_test_outputs.pkl", "rb") as f:
        hyps_2, refs_2 = pickle.load(f)

    metrics_2 = Metrics(hyps_2, refs_2)
    metrics_2 = metrics_2.get_statistics(
        stats={'chrFpp', 'meteor'}, 
        verbose=False, with_symbolic=False, bert=False
    )


    for method_1 in [
        "LaRAdam", "Adam", "LaAdam", "Adamax", "LaAdamax",
        "DiffGrad", "LaDiffGrad", "Lamb", "LaLamb", "SGD", "LaSGD", 
        "Yogi", "LaYogi"
    ]:
        print(method_1)
        with open(DIR + "/" + method_1 + "_test_outputs.pkl", "rb") as f:
            hyps_1, refs_1 = pickle.load(f)

        metrics_1 = Metrics(hyps_1, refs_1)
        metrics_1 = metrics_1.get_statistics(
            stats={'chrFpp', 'meteor'}, 
            verbose=False, with_symbolic=False, bert=False
        )

        if metrics_1['score'][metric] > metrics_2['score'][metric]:
            pc = round(100 * (metrics_1['score'][metric] - metrics_2['score'][metric]) / metrics_2['score'][metric], 2)
            print('>', f'({pc}%)', end=' ')
            h1 = 'less'
        else:
            pc = round(100 * (metrics_2['score'][metric] - metrics_1['score'][metric]) / metrics_2['score'][metric], 2)
            print('<', f'({pc}%)', end=' ')
            h1 = 'greater'
                    
        print()
        w, p = wilcoxon(
            metrics_1['scores'][metric],
            metrics_2['scores'][metric],
            alternative=h1
        )
        print('wilcoxon signed-rank: w =', round(int(w), -4), ', p =', p)
        if p > 0.05:
            if h1 == "less":
                print("failed to reject H0:", method_1, "<" if h1 == "greater" else ">", method_2, ", so")
                print("\033[93m" + method_1 + (" < " if h1 == "greater" else " > ") + method_2 + "\033[0m")
            else:
                print("failed to reject H0:", method_1, "<" if h1 == "greater" else ">", method_2, ", so")
                print("\033[92m" + method_1 + (" < " if h1 == "greater" else " > ") + method_2 + "\033[0m")
        else:
            print("reject H0:", method_1, "<" if h1 == "greater" else ">", method_2, ", so")
            print("\033[41m" + method_1 + (" > " if h1 == "greater" else " < ") + method_2 + "\033[0m")
    print()