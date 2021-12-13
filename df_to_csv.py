#!/usr/bin/env python3
# Copyright 2021 Dmitry Vilensky-Pasechnyuk

import pandas as pd
import pickle


file = "code2seq_pythonxglue_metrics"

with open(file + ".data", "rb") as f:
    df = pickle.load(f)

df.to_csv(file + ".csv", index=False)
