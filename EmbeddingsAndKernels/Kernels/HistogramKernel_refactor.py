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

HIST_KERNEL = np.dot(HISTOGRAMS, HISTOGRAMS.T)
normFactor = np.diag(HIST_KERNEL)
X, Y = np.meshgrid(normFactor, normFactor)
HIST_KERNEL = np.divide(HIST_KERNEL, np.sqrt(np.multiply(X, Y)))

HIST_DISTANCE = 1 - HIST_KERNEL

# Save the results as pkl
with open('data/distances/HIST_DISTANCE.pkl', 'wb') as f:
    pkl.dump(HIST_DISTANCE, f)
with open('data/distances/ORG_HIST_DISTANCE.pkl', 'wb') as f:
    pkl.dump(organism_names, f)

