import os
import numpy as np
import itertools
import multiprocessing
from collections import Counter
from joblib import Parallel, delayed
import pickle as pkl


with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]


organism_names = [data['ID'] for data in DATASET]

numGraphs = len(DATASET)

allSimplices = [simplex for data in DATASET for simplex in data['simplices_nodelabels']]
allSimplices = list(set(allSimplices))
d = {allSimplices[i]: i for i in range(len(allSimplices))}

HISTOGRAMS = np.zeros((numGraphs, len(allSimplices)))
for i in range(numGraphs):
    c = Counter(DATASET[i]['simplices_nodelabels'])
    for simplex, count in c.items():
        HISTOGRAMS[i, d[simplex]] = count

mask = HISTOGRAMS > 0
JACCARD_KERNEL = np.zeros((numGraphs, numGraphs))

def calculate_jaccard_row(i):
    OR = np.logical_or(mask[i, :], mask[i:, :])
    m = np.sum(np.minimum(HISTOGRAMS[i, :], HISTOGRAMS[i:, :] * OR), axis=1)
    M = np.sum(np.maximum(HISTOGRAMS[i, :], HISTOGRAMS[i:, :] * OR), axis=1)
    row = np.divide(m, M, out=np.zeros_like(m), where=M != 0)
    print("row", i, "done")
    return i, row

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores, verbose=50)(delayed(calculate_jaccard_row)(i) for i in range(numGraphs))

for i, row in results:
    JACCARD_KERNEL[i:, i] = row

JACCARD_KERNEL = JACCARD_KERNEL + JACCARD_KERNEL.T - np.diag(np.diag(JACCARD_KERNEL))

JACCARD_DISTANCE = 1 - JACCARD_KERNEL


# Save the results as pkl
with open('data/distances/JACCARD_DISTANCE.pkl', 'wb') as f:
    pkl.dump(JACCARD_DISTANCE, f)
with open('data/distances/ORG_JACCARD_DISTANCE.pkl', 'wb') as f:
    pkl.dump(organism_names, f)