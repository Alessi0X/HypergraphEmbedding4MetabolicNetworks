import pandas as pd
import numpy as np
import os
import time
import warnings
from tqdm import tqdm
import pickle as pkl
import hypergraphx as hgx
import networkx as nx
from hypergraphx.representations.projections import clique_projection

start = time.time()

#load dataset
with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

def compute_node_betweenness(i):
    tempnet = DATASET[i]["simplices_nodelabels"]
    orgname = DATASET[i]["ID"].upper()
    hnet = hgx.Hypergraph()
    hnet.add_edges(tempnet)
    cliquep = clique_projection(hnet, keep_isolated=True)
    centrality = nx.betweenness_centrality(cliquep, normalized=True)
    return orgname, centrality

from joblib import Parallel, delayed
num_cores = int(input("Number of cores to use: "))
assert num_cores > 0
assert num_cores <= os.cpu_count()

bagofwordsnodes = {}
results = Parallel(n_jobs=num_cores, verbose=50)(delayed(compute_node_betweenness)(i) for i in range(len(DATASET)))
for orgname, centrality in results:
    bagofwordsnodes.setdefault(orgname, {}).update(centrality)

df = pd.DataFrame.from_dict(bagofwordsnodes, orient="index").fillna(0)

df.to_csv("data/embeddings/NodeBetweenness.csv")
print("CSV SAVED")

print("...computing distance matrix...")

from scipy.spatial.distance import pdist, squareform

embeddingmatrix = df.values

distances = pdist(embeddingmatrix, metric="cityblock")

distance_matrix = squareform(distances)

# Save the distance matrix

with open("data/distances/BetCentManhattan.pkl", "wb") as f:
    pkl.dump(distance_matrix, f)
with open("data/distances/ORG_BetCentManhattan.pkl", "wb") as f:
    pkl.dump(df.index.tolist(), f)






