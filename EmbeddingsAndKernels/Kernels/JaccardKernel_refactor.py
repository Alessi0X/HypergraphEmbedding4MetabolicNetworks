import os
import time
import numpy as np
import pickle as pkl
import multiprocess as mp
from collections import Counter
from tqdm import tqdm

# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# Load dataset
with open("../../data/MetabolicPathways_DEMO_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

numGraphs = len(DATASET)

# set the number of cores for parallel processing
num_cores = os.cpu_count()
assert num_cores > 0
assert 0 < num_cores <= os.cpu_count()

organism_names = [data["ID"] for data in DATASET]

allSimplices = [simplex for data in DATASET for simplex in data["simplices_nodelabels"]]
allSimplices = list(set(allSimplices))
d = {allSimplices[i]: i for i in range(len(allSimplices))}

HISTOGRAMS = np.zeros((numGraphs, len(allSimplices)))
for i in range(numGraphs):
    c = Counter(DATASET[i]["simplices_nodelabels"])
    for simplex, count in c.items():
        HISTOGRAMS[i, d[simplex]] = count

mask = HISTOGRAMS > 0
JACCARD_KERNEL = np.zeros((numGraphs, numGraphs))

del DATASET  # housekeeping


# Helper function that computes one row of the Jaccard kernel matrix
# NOTE: we only compute the upper triangular part of the matrix (we will then symmetrize it)
def calculate_jaccard_row(i):
    OR = np.logical_or(mask[i, :], mask[i:, :])
    m = np.sum(np.minimum(HISTOGRAMS[i, :], HISTOGRAMS[i:, :] * OR), axis=1)
    M = np.sum(np.maximum(HISTOGRAMS[i, :], HISTOGRAMS[i:, :] * OR), axis=1)
    row = np.divide(m, M, out=np.zeros_like(m), where=M != 0)
    return i, row


# Use multiprocess.imap to parallelize the computation
with mp.Pool(processes=num_cores) as pool:
    results = pool.imap(calculate_jaccard_row, range(numGraphs))
    for i, row in tqdm(results, total=numGraphs, desc="Computing Jaccard kernel rows"):
        JACCARD_KERNEL[i:, i] = row

del HISTOGRAMS  # housekeeping
del mask  # housekeeping

# Symmetrize the Jaccard kernel matrix
JACCARD_KERNEL = JACCARD_KERNEL + JACCARD_KERNEL.T - np.diag(np.diag(JACCARD_KERNEL))

JACCARD_DISTANCE = 1 - JACCARD_KERNEL

# Get time elapsed for building kernel
print(f"Time elapsed [kernel computation]: {time.time() - start}")

# Save the results as pkl
with open("../../data/distances/JACCARD_DISTANCE.pkl", "wb") as f:
    pkl.dump(JACCARD_DISTANCE, f)
with open("../../data/distances/ORG_JACCARD_DISTANCE.pkl", "wb") as f:
    pkl.dump(organism_names, f)

print(f"Time elapsed [total]: {time.time() - start}")
