import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import networkx as nx


for clev in ["C1C2", "C3", "C4"]:
    with open(f"data/embeddings/NewGraph2Vec/{clev}_embedding.pkl", "rb") as f:
        embedding = pickle.load(f)

    assert clev == embedding["classlev"], "Class level mismatch in embedding data."

    orgs = embedding["orglist"]
    embdata = embedding["ebedding"]

    distancematrix = squareform(pdist(embdata, metric="euclidean"))
    print(f"distance matrix shape for class {clev} is: {distancematrix.shape}")

    with open(f"data/distances/temp/NewGraph2Vec{clev}Dist.pkl", "wb") as f:
        pickle.dump(distancematrix, f)
    with open(f"data/distances/temp/ORG_NewGraph2Vec{clev}Dist.pkl", "wb") as f:
        pickle.dump(orgs, f)

    print(f"Distances for {clev} saved successfully.")

