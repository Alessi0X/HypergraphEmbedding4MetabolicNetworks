import os
import sys
import time
import pickle
import pandas as pd
import networkx as nx
import hypergraphx as hgx
from hypergraphx.representations.projections import clique_projection
import numpy as np
from tqdm import tqdm

# Create folders to save embeddings
os.makedirs("../../data/embeddings/Graph2Vec", exist_ok=True)

# Add parent directories to Python path for custom imports
sys.path.insert(0, os.path.join(os.path.dirname(__name__), "..", ".."))
from customgraph2vec.mygraph2vec import MyGraph2Vec

# Set some useful parameters
nWorkers_Graph2Vec = os.cpu_count()  # Number of workers for Graph2Vec

# Load hypergraph dataset and build clique projections
with open("../../data/MetabolicPathways_DEMO_DATASET_Python.pkl", "rb") as f:
    DATASET = pickle.load(f)["DATASET"]
    for i in tqdm(range(len(DATASET)), desc="Evaluating clique projections (C1C2)"):
        tempnet = DATASET[i]["simplices_nodelabels"]
        orgname = DATASET[i]["ID"].upper()
        hnet = hgx.Hypergraph()
        hnet.add_edges(tempnet)
        cliquep = clique_projection(hnet, keep_isolated=False)
        DATASET[i] = {"ID": orgname, "Graph": cliquep}

# Extract lists
orglist = [DATASET[i]["ID"].upper() for i in range(len(DATASET))]
graphlist = [DATASET[i]["Graph"] for i in range(len(DATASET))]

# Prepare graphs (relabel nodes with integer ids + attribute)
prepgraphlist = []
for graph in tqdm(graphlist, desc="Preparing graphs for Graph2Vec (C1C2)"):
    for node in graph.nodes():
        graph.nodes[node]["feature"] = node
    mapping = {name: idx for idx, name in enumerate(graph.nodes())}
    temp = nx.relabel_nodes(graph, mapping)
    prepgraphlist.append(temp)

# Load labels
labels = pd.read_csv("../../data/organismsLabels.csv")
labels["Organism_shortname"] = labels["Organism_shortname"].fillna("NAN")
labels["Organism_shortname"] = labels["Organism_shortname"].str.upper()

# Filter to present organisms (no extra filtering for C1C2)
templabels = labels[labels["Organism_shortname"].isin(orglist)].copy(deep=True)

print("Running for class level: C1C2")
for c in ["C1", "C2"]:
    print(
        f"for {c} there are {len(prepgraphlist)} eligible graphs, {len(orglist)} eligible organisms and {templabels[c].nunique()} unique classes"
    )

# Load Optuna study and get the best parameters
with open("../../data/optuna_studies/graph2vecC1C2.pkl", "rb") as f:
    tempstudy = pickle.load(f)

tempparams = tempstudy.best_params
print(f"Best parameters for C1C2 are: {tempparams}")

# Start timer
start = time.time()
print("Computing Graph2Vec (C1C2)")

# Initialize and fit Graph2Vec model with best parameters
tempg2v = MyGraph2Vec(
    workers=nWorkers_Graph2Vec,
    dimensions=tempparams["dimensions"],
    wl_iterations=tempparams["wl_iterations"],
    epochs=tempparams["epochs"],
    learning_rate=tempparams["lr"],
    seed=472001,
    attributed=True,
    min_count=tempparams["mincount"],
)

tempg2v.fit(prepgraphlist)

# Get embeddings
tempemb = tempg2v.get_embedding()

print(f"Time elapsed [Graph2Vec fitting + embedding]: {time.time() - start}")
print(f"Embedding for class C1C2 shape is: {tempemb.shape}")

embtosave = {"classlev": "C1C2", "embedding": tempemb, "orglist": orglist}

with open("../../data/embeddings/Graph2Vec/C1C2_embedding.pkl", "wb") as f:
    pickle.dump(embtosave, f)
