import os
import time
import numpy as np
import pickle as pkl
from collections import Counter


# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# Load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

numGraphs = len(DATASET)

organism_names = [data["ID"] for data in DATASET]

allSimplices = [simplex for data in DATASET for simplex in data["simplices_nodelabels"]]
allSimplices = list(set(allSimplices))
d = {allSimplices[i]: i for i in range(len(allSimplices))}

HISTOGRAMS = np.zeros((numGraphs, len(allSimplices)))
for i in range(numGraphs):
    c = Counter(DATASET[i]["simplices_nodelabels"])
    for simplex, count in c.items():
        HISTOGRAMS[i, d[simplex]] = count

del DATASET  # housekeeping

HIST_KERNEL = np.dot(HISTOGRAMS, HISTOGRAMS.T)
normFactor = np.diag(HIST_KERNEL)
X, Y = np.meshgrid(normFactor, normFactor)
HIST_KERNEL = np.divide(HIST_KERNEL, np.sqrt(np.multiply(X, Y)))

del HISTOGRAMS  # housekeeping

HIST_DISTANCE = 1 - HIST_KERNEL

# Get time elapsed for building embedding
print(f"Time elapsed [kernel computation]: {time.time() - start}")

# Save the results as pkl
with open("../../data/distances/HIST_DISTANCE.pkl", "wb") as f:
    pkl.dump(HIST_DISTANCE, f)
with open("../../data/distances/ORG_HIST_DISTANCE.pkl", "wb") as f:
    pkl.dump(organism_names, f)

print(f"Time elapsed [total]: {time.time() - start}")
