import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl



#load dataset
with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

numGraphs = len(DATASET)

# BAG OF NODES EMBEDDING

# get set of organisms
organisms = [data['ID'] for data in DATASET]

# Create embeddings with a progress bar
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

#sort the embedding df acording to organisms
embeddingdf = embeddingdf.loc[organisms]


# save created embedding
embeddingdf.to_csv("data/embeddings/NodeDegree.csv")
print("Embedding saved")


print("...computing distance matrix...")
from scipy.spatial.distance import pdist, squareform

embeddingmatrix = embeddingdf.values

distancevector = pdist(embeddingmatrix, metric="euclidean")

distancematrix = squareform(distancevector)

with open("data/distances/NodeDegreeDistance.pkl", "wb") as f:
    pkl.dump(distancematrix, f)
with open("data/distances/ORG_NodeDegreeDistance.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)
print("distance matrix computed")
