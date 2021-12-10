#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import os
from textmetric import Metrics
import pandas as pd
from tqdm import tqdm 
import pickle
from pprint import pprint

for model_dir in filter(lambda dir: dir.startswith("results_"), os.listdir()):
    model = model_dir.split("_")[1]

    for dataset in os.listdir(model_dir):
        print("dataset:", dataset)

        data = []
        methods = []
        print("methods:")
        for method in os.listdir(os.path.join(model_dir, dataset)):
            methods.append(method.split("_")[0])
            print(methods[-1])

            with open(os.path.join(model_dir, dataset, method), "rb") as f:
                data.append(pickle.load(f))

        results = []
        for hyps, refs in tqdm(data):
            score = Metrics(hyps, refs).get_statistics()["score"]
            results.append(score)
            pprint(score)

        df = pd.DataFrame.from_records(results)
        df["Method"] = methods

        print(df)
        with open("_".join([model, dataset, "metrics"]) + ".data", "wb") as f:
            pickle.dump(df, f)
