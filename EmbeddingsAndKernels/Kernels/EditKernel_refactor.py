import os
import time
import numpy as np
import pickle as pkl
import multiprocess as mp
from tqdm import tqdm
from levenshteinDistance import levenshteinDistance

# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# Load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

numGraphs = len(DATASET)

# set the number of cores for parallel processing
num_cores = os.cpu_count()
assert num_cores > 0
assert 0 < num_cores <= os.cpu_count()

organism_names = [data["ID"] for data in DATASET]

for data in DATASET:
    data["simplices_nodelabels"].sort()  # lexicographic sort
    data["simplices_nodelabels"].sort(key=len)  # sort by ascending length


# Helper function that computes one row of the Edit kernel matrix
# NOTE: we only compute the upper triangular part of the matrix (we will then symmetrize it)
def levWrapperRow(i):
    graph_i = DATASET[i]["simplices_nodelabels"]
    # Only allocate memory for upper triangular part
    thisRow = np.zeros(numGraphs - i)

    for idx, j in enumerate(range(i, numGraphs)):
        graph_j = DATASET[j]["simplices_nodelabels"]
        GLD = levenshteinDistance(graph_i, graph_j)

        denominator = len(graph_i) + len(graph_j) + GLD
        NGLD = (2.0 * GLD) / denominator
        thisRow[idx] = 1.0 - NGLD
        # if denominator > 0:
        #     NGLD = (2.0 * GLD) / denominator
        #     thisRow[idx] = 1.0 - NGLD
        # else:
        #     thisRow[idx] = 1.0

    return i, thisRow


EDIT_KERNEL = np.zeros((numGraphs, numGraphs))

with mp.Pool(processes=num_cores) as pool:
    results = pool.imap(levWrapperRow, range(numGraphs))
    for i, row in tqdm(results, total=numGraphs, desc="Computing Edit kernel rows"):
        # Only assign to upper triangular part
        EDIT_KERNEL[i, i:] = row

del DATASET  # housekeeping

# Symmetrization using numpy indexing (more memory efficient than X + X.T - np.diag(np.diag(X)))
upper_tri_indices = np.triu_indices(numGraphs, k=1)
EDIT_KERNEL[upper_tri_indices[1], upper_tri_indices[0]] = EDIT_KERNEL[upper_tri_indices]

EDIT_DISTANCE = 1.0 - EDIT_KERNEL

# Get time elapsed for building kernel
print(f"Time elapsed [kernel computation]: {time.time() - start}")

# Save the results as pkl
with open("../../data/distances/EDIT_DISTANCE.pkl", "wb") as f:
    pkl.dump(EDIT_DISTANCE, f)
with open("../../data/distances/ORG_EDIT_DISTANCE.pkl", "wb") as f:
    pkl.dump(organism_names, f)

print(f"Time elapsed [total]: {time.time() - start}")
