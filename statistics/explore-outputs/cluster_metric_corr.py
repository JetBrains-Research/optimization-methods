#!/usr/bin/env python3

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
from textmetric.metrics import Metrics
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt


random_state = 0 
DIR = "outputs-javaxglue-code2seq"

cluster_score = "Calinski-Harabasz"
metric = "chrFpp"
clusters = 5
scores = {}
scores[metric] = []
cluster_scores = []

for method in [
    "Adamax", "LaAdamax", "DiffGrad", "LaDiffGrad", 
    "Adam", "LaRAdam", "Yogi", "LaYogi"
]:
    print(method)
    with open(DIR + "/" + method + "_test_outputs.pkl", "rb") as f:
        hyps, refs = pickle.load(f)

    metrics = Metrics(hyps, refs)
    metrics = metrics.get_statistics(
        stats={metric}, 
        verbose=False, with_symbolic=False, bert=False
    )

    scores[metric].append(metrics['score'][metric])
    print("scores", scores[metric][-1])

    df = pd.DataFrame(list(zip(hyps, refs)), columns=['hyp', 'text'])
    vec = TfidfVectorizer(stop_words="english")
    vec.fit(df.hyp.values)
    features = vec.transform(df.hyp.values)

    cls = MiniBatchKMeans(n_clusters=clusters, random_state=random_state)
    cls.fit(features)
    cls.predict(features)

    pca = PCA(n_components=2, random_state=random_state)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(cls.cluster_centers_)

    if not os.path.isfile("clusters_" + method + ".png"):
        plt.clf()
        plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
        plt.savefig("clusters_" + method + ".png")

    cluster_scores.append(calinski_harabasz_score(features.toarray(), labels=cls.predict(features)))
    print("cluster_scores", cluster_scores[-1])
    print()


r, p = stats.pearsonr(scores[metric], cluster_scores)
print("rho =", r, "p-value =", p)

if not os.path.isfile("correlation_" + metric + "_" + str(clusters) + ".png"):
    plt.clf()
    plt.scatter(scores[metric], cluster_scores, label="$\\rho = " + str(round(r, 2)) + "$")
    plt.grid(alpha=0.4)
    plt.xlabel(metric + " score")
    plt.ylabel(cluster_score + " score")
    plt.legend()
    plt.savefig("correlation_" + cluster_score + "_" + metric + "_" + str(clusters) + ".png")

# Silhouette 5: rho = -0.6162089986823289 p-value = 0.10376931863344162
# Silhouette 3: rho = -0.29269412749320806 p-value = 0.48173677293545275
# Calinski-Harabasz 5: rho = -0.49660224976489603 p-value = 0.21063103706191227
# У Adam'а высокие очки кластеризации, вместе с Adamax
