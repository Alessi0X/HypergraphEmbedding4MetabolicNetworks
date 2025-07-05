import pandas as pd
import numpy as np
import os
import pickle as pkl
import hypergraphx as hgx
import networkx as nx
from hypergraphx.representations.projections import clique_projection
from joblib import Parallel, delayed


# Load dataset
with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Function to compute PageRank for each graph
def compute_node_pagerank(i):
    tempnet = DATASET[i]["simplices_nodelabels"]
    orgname = DATASET[i]["ID"].upper()
    hnet = hgx.Hypergraph()
    hnet.add_edges(tempnet)
    cliquep = clique_projection(hnet, keep_isolated=False)
    pagerank = nx.pagerank(cliquep, max_iter=5000)
    return orgname, pagerank

num_cores = int(input("Number of cores to use: "))
assert 0 < num_cores <= os.cpu_count()

bagofwordsnodes = {}
results = Parallel(n_jobs=num_cores, verbose=50)(delayed(compute_node_pagerank)(i) for i in range(len(DATASET)))
for orgname, pagerank in results:
    bagofwordsnodes.setdefault(orgname, {}).update(pagerank)

# Convert to DataFrame
df = pd.DataFrame.from_dict(bagofwordsnodes, orient="index").fillna(0)

# Save to CSV
df.to_csv("data/embeddings/NodePageRank.csv")
print("CSV SAVED")

print("...computing distance matrix...")

from scipy.spatial.distance import pdist, squareform

embeddingmatrix = df.values

distances = pdist(embeddingmatrix, metric="cityblock")

distance_matrix = squareform(distances)

# Save the distance matrix
with open("data/distances/PageRankManhattan.pkl", "wb") as f:
    pkl.dump(distance_matrix, f)

with open("data/distances/ORG_PageRankManhattan.pkl", "wb") as f:
    pkl.dump(df.index.tolist(), f)
