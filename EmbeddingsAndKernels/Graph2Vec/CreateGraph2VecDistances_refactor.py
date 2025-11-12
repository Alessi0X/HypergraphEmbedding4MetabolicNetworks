import os
import pickle
from scipy.spatial.distance import pdist, squareform


# Create folders to save distance matrices
os.makedirs("../../data/distances/Graph2Vec", exist_ok=True)

for clev in ["C1C2", "C3", "C4"]:
    with open(f"../../data/embeddings/Graph2Vec/{clev}_embedding.pkl", "rb") as f:
        embedding = pickle.load(f)

    assert clev == embedding["classlev"], "Class level mismatch in embedding data."

    orgs = embedding["orglist"]
    embdata = embedding["embedding"]

    distancematrix = squareform(pdist(embdata, metric="euclidean"))
    print(f"Distance matrix shape for class {clev} is: {distancematrix.shape}")

    with open(f"../../data/distances/Graph2Vec/Graph2Vec_{clev}_Dist.pkl", "wb") as f:
        pickle.dump(distancematrix, f)
    with open(
        f"../../data/distances/Graph2Vec/ORG_Graph2Vec_{clev}_Dist.pkl", "wb"
    ) as f:
        pickle.dump(orgs, f)

    print(f"Distances for {clev} saved successfully.\n")
