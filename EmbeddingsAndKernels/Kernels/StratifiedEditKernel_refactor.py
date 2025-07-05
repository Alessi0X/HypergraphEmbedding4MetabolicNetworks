import os
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from levenshteinDistance import levenshteinDistance
import itertools
import pickle as pkl

with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

numGraphs = len(DATASET)

for data in DATASET:
    d = {k: sorted(list(g)) for k, g in itertools.groupby(sorted(data['simplices_nodelabels'], key=lambda x: x.count(',')), key=lambda x: x.count(','))}
    data['simplices_groupByOrder'] = d

def stratifiedLevWrapperRow(i):
    graph_i = DATASET[i]['simplices_groupByOrder']
    thisRow = [0.0] * numGraphs
    for j in range(i, numGraphs):
        graph_j = DATASET[j]['simplices_groupByOrder']
        orders = set(list(graph_i.keys()) + list(graph_j.keys()))
        similarities = []
        for k in orders:
            subset_i = graph_i.get(k, [])
            subset_j = graph_j.get(k, [])
            GLD = levenshteinDistance(subset_i, subset_j)
            NGLD = float(2 * GLD) / float(len(subset_i) + len(subset_j) + GLD) if (len(subset_i)+len(subset_j)+GLD)!=0 else 0
            similarities.append(1 - NGLD)
        thisRow[j] = np.mean(similarities)
    return i, thisRow

num_cores = multiprocessing.cpu_count()

tmp = Parallel(n_jobs=num_cores, verbose=50)(delayed(stratifiedLevWrapperRow)(i) for i in range(numGraphs))
STRATEDIT_KERNEL = np.empty((numGraphs, numGraphs))
for i, row in tmp:
    STRATEDIT_KERNEL[i, :] = row

STRATEDIT_KERNEL = STRATEDIT_KERNEL + STRATEDIT_KERNEL.T - np.diag(np.diag(STRATEDIT_KERNEL))

STRATEDIT_DISTANCE = 1 - STRATEDIT_KERNEL

organism_names = [data['ID'] for data in DATASET]

# Save the results as pkl
with open('data/distances/STRATEDIT_DISTANCE.pkl', 'wb') as f:
    pkl.dump(STRATEDIT_DISTANCE, f)
with open('data/distances/ORG_STRATEDIT_DISTANCE.pkl', 'wb') as f:
    pkl.dump(organism_names, f)

