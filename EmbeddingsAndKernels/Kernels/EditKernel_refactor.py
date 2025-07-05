import os
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from levenshteinDistance import levenshteinDistance
import pickle as pkl

with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

organism_names = [data['ID'] for data in DATASET]
numGraphs = len(DATASET)

for data in DATASET:
    data['simplices_nodelabels'].sort()                        # lexicographic sort
    data['simplices_nodelabels'].sort(key=len)  # sort by ascending length

def levWrapperRow(i):
    graph_i = DATASET[i]['simplices_nodelabels']
    thisRow = [0.0] * numGraphs
    for j in range(i, numGraphs):
        graph_j = DATASET[j]['simplices_nodelabels']
        GLD = levenshteinDistance(graph_i, graph_j)
        NGLD = float(2 * GLD) / float(len(graph_i) + len(graph_j) + GLD)
        thisRow[j] = 1 - NGLD
    return i, thisRow

num_cores = multiprocessing.cpu_count()

tmp = Parallel(n_jobs=num_cores, verbose=50)(delayed(levWrapperRow)(i) for i in range(numGraphs))
EDIT_KERNEL = np.empty((numGraphs, numGraphs))
for i, row in tmp:
    EDIT_KERNEL[i, :] = row
EDIT_KERNEL = EDIT_KERNEL + EDIT_KERNEL.T - np.diag(np.diag(EDIT_KERNEL))


EDIT_DISTANCE = 1 - EDIT_KERNEL

# Save the results as pkl
with open('data/distances/EDIT_DISTANCE.pkl', 'wb') as f:
    pkl.dump(EDIT_DISTANCE, f)
with open('data/distances/ORG_EDIT_DISTANCE.pkl', 'wb') as f:
    pkl.dump(organism_names, f)


