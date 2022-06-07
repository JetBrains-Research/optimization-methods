#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from textmetric.metrics import Metrics
import matplotlib.pyplot as plt


random_state = 0 

DIR = "outputs-javaxglue-code2seq"
method = "Adam"
metric = "chrFpp"
clusters = 3

with open(DIR + "/" + method + "_test_outputs.pkl", "rb") as f:
    hyps, refs = pickle.load(f)

metrics = Metrics(hyps, refs)
metrics = metrics.get_statistics(
    stats={metric}, 
    verbose=False, with_symbolic=False, bert=False
)

df = pd.DataFrame(list(zip(hyps, refs)), columns=['hyp', 'text'])
vec = TfidfVectorizer(stop_words="english")
vec.fit(df.hyp.values)
features = vec.transform(df.hyp.values)

cls = MiniBatchKMeans(n_clusters=clusters, random_state=random_state)
cls.fit(features)
cls.predict(features)

result = pd.DataFrame(list(zip(metrics['scores'][metric], cls.predict(features))), columns=['scores', 'class'])
print("Mean groupwise score for hyps clustering:")
print(result.groupby('class')['scores'].mean())


vec = TfidfVectorizer(stop_words="english")
vec.fit(df.text.values)
features = vec.transform(df.text.values)

cls = MiniBatchKMeans(n_clusters=clusters, random_state=random_state)
cls.fit(features)
cls.predict(features)

result = pd.DataFrame(list(zip(metrics['scores'][metric], cls.predict(features))), columns=['scores', 'class'])
print()
print("Mean groupwise score for refs clustering:")
print(result.groupby('class')['scores'].mean())
