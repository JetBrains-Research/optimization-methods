#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_files


random_state = 0 

DIR = "outputs-javaxglue-code2seq"
method = "Adam"

with open(DIR + "/" + method + "_test_outputs.pkl", "rb") as f:
    hyps, refs = pickle.load(f)

df = pd.DataFrame(list(zip(hyps, refs)), columns=['hyp', 'text'])

vec = TfidfVectorizer(stop_words="english")
vec.fit(df.text.values)
features = vec.transform(df.text.values)

cls = MiniBatchKMeans(n_clusters=3, random_state=random_state)
cls.fit(features)
cls.predict(features)

if not os.path.isfile("clusters_pca.png"):
    pca = PCA(n_components=2, random_state=random_state)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(cls.cluster_centers_)

    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    plt.savefig("clusters_pca.png")

result = pd.DataFrame(list(zip(refs, cls.predict(features))), columns=['text', 'class'])

print("Grouping refs:")
print(result.groupby('class').sample(frac=.005))


vec = TfidfVectorizer(stop_words="english")
vec.fit(df.hyp.values)
features = vec.transform(df.hyp.values)

cls = MiniBatchKMeans(n_clusters=3, random_state=random_state)
cls.fit(features)
cls.predict(features)

if not os.path.isfile("clusters_pca_hyp.png"):
    pca = PCA(n_components=2, random_state=random_state)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(cls.cluster_centers_)

    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    plt.savefig("clusters_pca_hyp.png")

result_2 = pd.DataFrame(list(zip(hyps, cls.predict(features))), columns=['hyp', 'class'])
print()
print("Grouping hyps:")
print(result_2.groupby('class').sample(frac=.004))

result_2 = pd.DataFrame(list(zip(refs, hyps, cls.predict(features))), columns=['text', 'hyp', 'class'])


def jaccard(list1, list2):
    cf = len(set(list1).intersection(list2)) + 0.
    df1 = len(set(list1) - set(list2)) + 0.
    df2 = len(set(list2) - set(list1)) + 0.
    return cf / (cf + df1) / (cf + df2) / (
        df1 / (cf + df1) + df2 / (cf + df2) + cf / (cf + df1) / (cf + df2)
    )


print()
for i in range(3):
    for j in range(3):
        l1 = result_2.where(result_2["class"] == i)["text"].tolist()
        l2 = result.where(result["class"] == j)["text"].tolist()
        print("Hyp class", i, "and Ref class", j, "are similar with J_N =", round(jaccard(l1, l2) * 10**5, 1))
    print()
