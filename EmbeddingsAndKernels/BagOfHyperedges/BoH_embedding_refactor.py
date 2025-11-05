import os
import time
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# Load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

numGraphs = len(DATASET)

# Get set of organisms
organisms = [data["ID"] for data in DATASET]

# Create embeddings (with a progress bar)
bagofwords_dict = {}

for idx in tqdm(range(numGraphs), desc="Processing organisms"):
    content = DATASET[idx]["simplices_nodelabels"]
    org = DATASET[idx]["ID"].upper()
    content = [",".join(sorted(x)) for x in content]
    for hyperedge in content:
        bagofwords_dict.setdefault(org, {})[hyperedge] = 1

del DATASET  # housekeeping

# Convert dict to DataFrame
embeddingdf = pd.DataFrame.from_dict(bagofwords_dict, orient="index").fillna(0)

del bagofwords_dict  # housekeeping

# Check if embedding df has the same elements as organisms (...just to be sure)
organisms = [x.upper() for x in organisms]
assert set(embeddingdf.index) == set(organisms)

# Sort the embedding df according to organisms
embeddingdf = embeddingdf.loc[organisms]

# Get time elapsed for building embedding
print(f"Time elapsed [embedding only]: {time.time() - start}")

# Save created embedding
print("Saving embedding with shape: ", embeddingdf.shape)
embeddingdf.to_csv("../../data/embeddings/BagOfHyperedges.csv")

# Create distance matrices
print("Calculating distance matrices")

# Extract
embeddingmatrix = embeddingdf.values

# Calculate distance matrix
distmatrix = pdist(embeddingmatrix, metric="jaccard")
distmatrix = squareform(distmatrix)

# Save distance matrix
with open("../../data/distances/BagOfHyperedgesJaccard.pkl", "wb") as f:
    pkl.dump(distmatrix, f)
with open("../../data/distances/ORG_BagOfHyperedgesJaccard.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)

distmatrixman = pdist(embeddingmatrix, metric="cityblock")
distmatrixman = squareform(distmatrixman)

# Save distance matrix
with open("../../data/distances/BagOfHyperedgesManhattan.pkl", "wb") as f:
    pkl.dump(distmatrixman, f)
with open("../../data/distances/ORG_BagOfHyperedgesManhattan.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)

# that's all folks
print(f"Time elapsed [embedding + distance matrix]: {time.time() - start}")
