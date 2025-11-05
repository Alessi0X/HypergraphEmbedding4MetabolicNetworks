import os
import time
import pandas as pd
import pickle as pkl
import hypergraphx as hgx
import networkx as nx
import multiprocess as mp
from tqdm import tqdm
from hypergraphx.representations.projections import clique_projection
from scipy.spatial.distance import pdist, squareform


# Create folders to save embeddings and distance matrices
os.makedirs("../../data/distances", exist_ok=True)
os.makedirs("../../data/embeddings", exist_ok=True)

# Load dataset
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

# Start timer
start = time.time()

# set the number of cores for parallel processing
num_cores = os.cpu_count()
assert num_cores > 0
assert 0 < num_cores <= os.cpu_count()

numGraphs = len(DATASET)


# Helper function to compute PageRank for each graph
def compute_node_pagerank(i):
    tempnet = DATASET[i]["simplices_nodelabels"]
    orgname = DATASET[i]["ID"].upper()
    hnet = hgx.Hypergraph()
    hnet.add_edges(tempnet)
    cliquep = clique_projection(hnet, keep_isolated=False)
    pagerank = nx.pagerank(cliquep, max_iter=5000)
    return orgname, pagerank


bagofwordsedges = {}

# Use multiprocess.imap to parallelize the computation
with mp.Pool(processes=num_cores) as pool:
    results = pool.imap(compute_node_pagerank, range(numGraphs))
    for orgname, pagerank in tqdm(results, total=numGraphs, desc="Computing PageRank"):
        bagofwordsedges.setdefault(orgname, {}).update(pagerank)

del DATASET  # housekeeping

# Convert to DataFrame
embeddingdf = pd.DataFrame.from_dict(bagofwordsedges, orient="index").fillna(0)

del bagofwordsedges  # housekeeping

# Get time elapsed for building embedding
print(f"Time elapsed [embedding only]: {time.time() - start}")

# Save created embedding
print("Saving embedding with shape: ", embeddingdf.shape)
embeddingdf.to_csv("../../data/embeddings/NodePageRank.csv")

# Create distance matrices
print("Calculating distance matrices")

embeddingmatrix = embeddingdf.values

distances = pdist(embeddingmatrix, metric="cityblock")
distance_matrix = squareform(distances)

# Save distance matrix
with open("../../data/distances/PageRankManhattan.pkl", "wb") as f:
    pkl.dump(distance_matrix, f)
with open("../../data/distances/ORG_PageRankManhattan.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)

# that's all folks
print(f"Time elapsed [embedding + distance matrix]: {time.time() - start}")
