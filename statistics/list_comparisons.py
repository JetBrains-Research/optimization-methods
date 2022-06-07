from cgi import print_directory
import os
import pandas as pd
from tqdm import tqdm 
import pickle
import random
import numpy as np
from scipy.stats import wilcoxon
import sys
sys.path.append("..")


launch_id = "results_ct-pythonxglue"

with open(f"dump_comparisons_{launch_id}.data", "rb") as file:
    method_names, comparisons = pickle.load(file)

print(method_names)

for i, method_a in enumerate(method_names[:-1]):
    for j, method_b in enumerate(method_names[i+1:]):
        if (comparisons[i][j].find("[>]") != -1) or (comparisons[i][j].find("[?]") != -1) or not comparisons[i][j].endswith("100.0%"):
            print(method_a, method_b)
            print(comparisons[i][j])
            print()
    print()