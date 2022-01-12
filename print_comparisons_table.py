#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import numpy as np
import pandas as pd
import pickle
from pprint import pprint


folder = "code2seq-javaxglue"
docstring = True


with open(f"dump_comparisons_results_{folder}.data", "rb") as file:
    method_names, comparisons = pickle.load(file)

with open(f"dump_for_tables_results_{folder}.data", "rb") as file:
    if docstring:
        meteor_for_method, blue_for_method = pickle.load(file)
    else:
        F1_for_method, score = pickle.load(file)

table = pd.DataFrame()
table['Method'] = method_names
if docstring:
    table['meteor'] = [meteor_for_method[method] for method in method_names]
    table['bleu'] = [blue_for_method[method] for method in method_names]
    
    print(table)

for i, method_a in enumerate(method_names[:-1]):
    for j, method_b in enumerate(method_names[i+1:]):
        print(method_a, method_b)
        print(comparisons[i][j])
        print()
        
    print()
    print()