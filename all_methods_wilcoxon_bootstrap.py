#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import os
from textmetric import Metrics
import pandas as pd
from tqdm import tqdm 
import pickle
import random
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_compare(scores_a, scores_b, h0, resamples=1000):
    sampled_scores_pairs = []
    for _ in range(resamples):
        # k doubtful, but okay https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
        indices = random.choices(range(scores_a.shape[0]), k=scores_a.shape[0])
        aggregated_a = np.mean(scores_a[indices])
        aggregated_b = np.mean(scores_b[indices])
        sampled_scores_pairs.append((aggregated_a, aggregated_b))

    score = 0.
    for aggregated_a, aggregated_b in sampled_scores_pairs:
        if (h0 == '>' and aggregated_a > aggregated_b) or (h0 == '<' and aggregated_a < aggregated_b):
            score += 1
    return score / len(sampled_scores_pairs)


tasks = [
    {"folder": "results_treelstm/javaxglue", "docstring": True, "compare": False}
]


for task in tasks:

    folder = task["folder"]
    docstring = task["docstring"]
    compare = task["compare"]

    print('---------------------')
    print(folder)
    print('---------------------\n')

    launch_id = folder.replace("/", "-")

    if os.path.isfile(f"dump_for_wilcoxon_bootstrap_{launch_id}.data"):
        with open(f"dump_for_wilcoxon_bootstrap_{launch_id}.data", "rb") as file:
            method_names, score, scores_for_method = pickle.load(file)
    else:
        hyps_ref_all_methods = []
        method_names = []
        outputs_for_method = {}

        with open(os.path.join(folder, "refs.pkl"), "rb") as file:
            refs = pickle.load(file)

        for outputs_file_name in os.listdir(folder):
            method_name = outputs_file_name.split("_")[0]
            method_names.append(method_name)
            
            with open(os.path.join(folder, outputs_file_name), "rb") as file:
                outputs_for_method[method_name] = pickle.load(file), refs

        if docstring:
            meteor_for_method = {}
            blue_for_method = {}
            F1_for_method = {}
        else:
            F1_for_method = {}

        score = {}
        scores_for_method = {}

        print("Calculating metrics...")

        for method_name in tqdm(method_names):
            print(method_name)
            hyps, refs = outputs_for_method[method_name]
            
            if docstring:
                evaluated_metrics = Metrics(hyps, refs).get_statistics(stats={"chrF", "meteor", "bleu", "prec_rec_f1"})
                meteor_for_method[method_name] = evaluated_metrics["score"]["meteor"]
                blue_for_method[method_name] = evaluated_metrics["score"]["bleu"]
                F1_for_method[method_name] = evaluated_metrics["score"]["f1"]
                scores_for_method[method_name] = evaluated_metrics["scores"]["chrF"]
                score[method_name] = evaluated_metrics["score"]["chrF"]
            else:
                evaluated_metrics = Metrics(hyps, refs).get_statistics(stats={"chrF", "prec_rec_f1"})
                F1_for_method[method_name] = evaluated_metrics["score"]["f1"]
                scores_for_method[method_name] = evaluated_metrics["scores"]["f1"]
                score[method_name] = evaluated_metrics["score"]["chrF"]

        print("Metrics are calculated.")

        with open(f"dump_for_wilcoxon_bootstrap_{launch_id}.data", "wb") as file:
            pickle.dump((method_names, score, scores_for_method), file)

        with open(f"dump_for_tables_{launch_id}.data", "wb") as file:
            if docstring:
                pickle.dump((meteor_for_method, blue_for_method, score, F1_for_method), file)
            else:
                pickle.dump((score, F1_for_method), file)

    if not compare:
        method_names.sort(key=lambda method: score[method])
        with open(f"dump_comparisons_{launch_id}.data", "wb") as file:
            pickle.dump(method_names, file)

    if compare:
        comparisons = [[f"" for _ in range(len(method_names))] for _ in range(len(method_names))]
        method_names.sort(key=lambda method: score[method])

        print("Comparing methods...")

        for i, method_a in tqdm(enumerate(method_names)):
            if folder[-7:] in ["javamed", "amed0.1"]:
                if len(scores_for_method[method_a]) == 417821:
                    scores_for_method[method_a] = np.append(scores_for_method[method_a], 0.0)
                    scores_for_method[method_a] = np.append(scores_for_method[method_a], 0.0)
                    scores_for_method[method_a] = np.append(scores_for_method[method_a], 0.0)
                    scores_for_method[method_a] = np.append(scores_for_method[method_a], 0.0)
            print(len(scores_for_method[method_a]))

        for i, method_a in tqdm(enumerate(method_names[:-1])):
            print(method_a)
            for j, method_b in enumerate(method_names[i+1:]):
                result = ""

                if score[method_a] > score[method_b]:
                    pc = round(100 * (score[method_a] - score[method_b]) / score[method_b], 2)
                    h0 = '>'
                    result += f"{h0} ({pc}%), "
                else:
                    pc = round(100 * (score[method_b] - score[method_a]) / score[method_b], 2)
                    h0 = '<'
                    result += f"{h0} ({pc}%), "
                
                h1 = '<' if h0 == '>' else '>'

                w, p = wilcoxon(scores_for_method[method_a], scores_for_method[method_b], alternative="less" if h0 == '>' else "greater")
                result += f"wilc. h0 1-p={1-p:.2E}: "

                if p > 0.05:
                    result += f"h0 not rej...\n"
                else:
                    result += f"h0 rej. [{h1}]"

                bootstrap_score_h0 = round(100 * bootstrap_compare(scores_for_method[method_a], scores_for_method[method_b], h0), 2)

                if p > 0.05:
                    w, p = wilcoxon(scores_for_method[method_a], scores_for_method[method_b], alternative="less" if h1 == '>' else "greater")
                    result += f"wilc. h1 1-p={1-p:.2E}: "

                    if p > 0.05:
                        result += f"h1 not rej. [?]"
                    else:
                        result += f"h1 rej. [{h0}]"

                    bootstrap_score_h1 = round(100 * bootstrap_compare(scores_for_method[method_a], scores_for_method[method_b], h1), 2)
                    result += f"\nbootstrap {h1} score = {bootstrap_score_h1}%"

                result += f"\nbootstrap {h0} score = {bootstrap_score_h0}%"
                comparisons[i][j] = result

        print("Comparison done.")

        with open(f"dump_comparisons_{launch_id}.data", "wb") as file:
            pickle.dump((method_names, comparisons), file)
