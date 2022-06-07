#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import numpy as np
import pandas as pd
import pickle
from pprint import pprint

for folder in ["treelstm-javaxglue"]:
    docstring = True
    compare = False


    if compare:
        with open(f"dump_comparisons_results_{folder}.data", "rb") as file:
            method_names, comparisons = pickle.load(file)
    else:
        with open(f"dump_comparisons_results_{folder}.data", "rb") as file:
            method_names = pickle.load(file)

    with open(f"dump_for_tables_results_{folder}.data", "rb") as file:
        if docstring:
            meteor_for_method, blue_for_method, chrF_for_method, F1_for_method = pickle.load(file)
        else:
            chrF_for_method, F1_for_method = pickle.load(file)

    table = pd.DataFrame()
    table['Method'] = method_names
    if docstring:
        table['chrF'] = [chrF_for_method[method] for method in method_names]
        table['f1'] = [F1_for_method[method] for method in method_names]
        table['meteor'] = [meteor_for_method[method] for method in method_names]
        table['bleu'] = [blue_for_method[method] for method in method_names]
        
        print(table)
    else:
        table['chrF'] = [chrF_for_method[method] for method in method_names]
        table['f1'] = [F1_for_method[method] for method in method_names]
        
        print(table)


# for i, method_a in enumerate(method_names[:-1]):
#     for j, method_b in enumerate(method_names[i+1:]):
#         if (comparisons[i][j].find("[>]") != -1) or (comparisons[i][j].find("[?]") != -1) or not comparisons[i][j].endswith("100.0%"):
#             print(method_a, method_b)
#             print(comparisons[i][j])
#             print()
#     print()