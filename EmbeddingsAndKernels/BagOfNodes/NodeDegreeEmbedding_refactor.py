import os
import time
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform


# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]
numGraphs = len(DATASET)

# Start timer
start = time.time()

# get set of organisms
organisms = [data["ID"] for data in DATASET]

# Create embeddings (with a progress bar)
bagofwords_dict = {}

for idx in tqdm(range(numGraphs), desc="Processing Files"):
    content = DATASET[idx]["simplices_nodelabels"]
    org = DATASET[idx]["ID"].upper()
    for simplex in content:
        for node in simplex:
            bagofwords_dict.setdefault(org, {}).setdefault(node, 0)
            bagofwords_dict[org][node] += 1

# Convert dict to DataFrame
embeddingdf = pd.DataFrame.from_dict(bagofwords_dict, orient="index").fillna(0)

organisms = [x.upper() for x in organisms]

# sort the embedding df according to organisms
embeddingdf = embeddingdf.loc[organisms]

# Get time elapsed for building embedding
print(f"Time elapsed [embedding only]: {time.time() - start}")

# save created embedding
print("Saving embedding with shape: ", embeddingdf.shape)
embeddingdf.to_csv("../../data/embeddings/NodeDegree.csv")

# create distance matrices
# print("Calculating distance matrices")
#
# embeddingmatrix = embeddingdf.values
#
# distancevector = pdist(embeddingmatrix, metric="euclidean")
# distancematrix = squareform(distancevector)
#
# with open("data/distances/NodeDegreeDistance.pkl", "wb") as f:
#     pkl.dump(distancematrix, f)
# with open("data/distances/ORG_NodeDegreeDistance.pkl", "wb") as f:
#     pkl.dump(embeddingdf.index.tolist(), f)
# print("distance matrix computed")

# that's all folks
print(f"Time elapsed [embedding + distance matrix]: {time.time() - start}")
