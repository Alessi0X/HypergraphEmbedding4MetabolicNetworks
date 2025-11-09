import os
import time
import numpy as np
import pickle as pkl
import itertools
import multiprocess as mp
from tqdm import tqdm
from levenshteinDistance import levenshteinDistance

# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# set the number of cores for parallel processing
num_cores = os.cpu_count()
assert num_cores > 0
assert 0 < num_cores <= os.cpu_count()

# Load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

numGraphs = len(DATASET)

organism_names = [data["ID"] for data in DATASET]

# Process and add stratified grouping to each dataset entry
for data in DATASET:
    d = {
        k: sorted(list(g))
        for k, g in itertools.groupby(
            sorted(data["simplices_nodelabels"], key=lambda x: x.count(",")),
            key=lambda x: x.count(","),
        )
    }
    data["simplices_groupByOrder"] = d


# Helper function that computes one row of the (stratified) Edit kernel matrix
# NOTE: we only compute the upper triangular part of the matrix (we will then symmetrize it)
def stratifiedLevWrapperRow(i):
    graph_i = DATASET[i]["simplices_groupByOrder"]
    # Only allocate memory for upper triangular part
    thisRow = np.zeros(numGraphs - i)

    for idx, j in enumerate(range(i, numGraphs)):
        graph_j = DATASET[j]["simplices_groupByOrder"]

        # Find common orders between the two simplicial complexes
        orders = set(graph_i.keys()) | set(graph_j.keys())

        if not orders:  # Handle empty case
            # Complexes with no simplicial order in common have zero similarity
            thisRow[idx] = 0.0
            continue

        similarities = []

        for k in orders:
            subset_i = graph_i.get(k, [])
            subset_j = graph_j.get(k, [])

            # Skip empty subsets to avoid unnecessary computation
            if not subset_i and not subset_j:
                similarities.append(1.0)  # Empty sets are identical
                continue

            GLD = levenshteinDistance(subset_i, subset_j)

            denominator = len(subset_i) + len(subset_j) + GLD
            NGLD = (2.0 * GLD) / denominator
            similarities.append(1.0 - NGLD)

        thisRow[idx] = np.mean(similarities)

    return i, thisRow


STRATEDIT_KERNEL = np.zeros((numGraphs, numGraphs))

with mp.Pool(processes=num_cores) as pool:
    results = pool.imap(stratifiedLevWrapperRow, range(numGraphs))
    for i, row in tqdm(
        results, total=numGraphs, desc="Computing Stratified Edit kernel rows"
    ):
        # Only fill upper triangular part
        STRATEDIT_KERNEL[i, i:] = row

del DATASET  # housekeeping

# Symmetrization using numpy indexing (more memory efficient than X + X.T - np.diag(np.diag(X)))
print("Symmetrizing kernel matrix...")
upper_tri_indices = np.triu_indices(numGraphs, k=1)
STRATEDIT_KERNEL[upper_tri_indices[1], upper_tri_indices[0]] = STRATEDIT_KERNEL[
    upper_tri_indices
]

STRATEDIT_DISTANCE = 1.0 - STRATEDIT_KERNEL

# Get time elapsed for building kernel
print(f"Time elapsed [kernel computation]: {time.time() - start}")

# Save the results with compression
with open("../../data/distances/STRATEDIT_DISTANCE.pkl", "wb") as f:
    pkl.dump(STRATEDIT_DISTANCE, f)
with open("../../data/distances/ORG_STRATEDIT_DISTANCE.pkl", "wb") as f:
    pkl.dump(organism_names, f)

print(f"Time elapsed [total]: {time.time() - start}")
