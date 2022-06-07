#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import os
from textmetric import Metrics
import pandas as pd
from tqdm import tqdm 
import pickle
from pprint import pprint
import numpy as np


model = "ct"
model_dir = "results_" + model
dataset = "javamed"

data = []
methods = []
print("methods:")
for method in os.listdir(os.path.join(model_dir, dataset)):

    if method == "analysis":
        continue

    methods.append(method.split("_")[0])
    print(methods[-1])

    with open(os.path.join(model_dir, dataset, method), "rb") as f:
        data.append(pickle.load(f))

results = []
for hyps, refs in tqdm(data):
    print(hyps[:10], refs[:10])
    res = Metrics(hyps, refs).get_statistics(stats={"prec_rec_f1", "chrF"}, bert=True)
    score = res["score"]
    scores = res["scores"]
    results.append(score)
    pprint(score)

df = pd.DataFrame.from_records(results)
df["Method"] = methods

print(df)
with open("_".join([model, dataset, "metrics"]) + ".data", "wb") as f:
    pickle.dump(df, f)
