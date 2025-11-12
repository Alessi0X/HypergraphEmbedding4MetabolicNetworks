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

# Add parent directory to Python path for custom imports (same style as original script)
sys.path.insert(0, os.path.join(os.path.dirname(__name__), "..", ".."))
from customgraph2vec.mygraph2vec import MyGraph2Vec

# Set some useful parameters
nWorkers_Graph2Vec = os.cpu_count()  # Number of workers for Graph2Vec

# Load hypergraph dataset and build clique projections
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pickle.load(f)["DATASET"]
    for i in tqdm(range(len(DATASET)), desc="Evaluating clique projections (C4)"):
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
for graph in tqdm(graphlist, desc="Preparing graphs for Graph2Vec (C4)"):
    for node in graph.nodes():
        graph.nodes[node]["feature"] = node
    mapping = {name: idx for idx, name in enumerate(graph.nodes())}
    temp = nx.relabel_nodes(graph, mapping)
    prepgraphlist.append(temp)

# Load labels
labels = pd.read_csv("../../data/organismsLabels.csv")
labels["Organism_shortname"] = labels["Organism_shortname"].fillna("NAN")
labels["Organism_shortname"] = labels["Organism_shortname"].str.upper()

# Filtering for C4: keep only labels with minocc >= 20, and drop organisms missing a C4 label
clev = "C4"
templabels = labels[labels["Organism_shortname"].isin(orglist)].copy(deep=True)
minocc = 20
count = labels[clev].value_counts().reset_index()
count.columns = [clev, "count"]
toremove = count[count["count"] < minocc][clev].tolist()
templabels = templabels[~(templabels[clev].isin(toremove))]

# Remove organisms lacking a C4 label
missinglist = templabels.loc[(templabels[clev].isna()), "Organism_shortname"].tolist()
templabels = templabels[~(templabels["Organism_shortname"].isin(missinglist))]

# Build filtered list of organisms and graphs
temporgs = templabels["Organism_shortname"].values

# Cut prepgraphlist accordingly
filtered_graphs = []
orglist_np = np.array(orglist)
for org in temporgs:
    idx = np.where(orglist_np == org)[0][0]
    filtered_graphs.append(prepgraphlist[idx])

print("Running for class level: C4")
print(
    f"for C4 there are {len(filtered_graphs)} eligible graphs, {len(temporgs)} eligible organisms and {templabels[clev].nunique()} unique classes"
)

# Load Optuna study and get the best parameters
with open("../../data/optuna_studies/graph2vecC4.pkl", "rb") as f:
    tempstudy = pickle.load(f)

tempparams = tempstudy.best_params
print(f"Best parameters for C4 are: {tempparams}")

# Start timer
start = time.time()
print("Computing Graph2Vec (C4)")

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

tempg2v.fit(filtered_graphs)

# Get embeddings
tempemb = tempg2v.get_embedding()

print(f"Time elapsed [Graph2Vec fitting + embedding]: {time.time() - start}")
print(f"Embedding for class C4 shape is: {tempemb.shape}")

embtosave = {"classlev": "C4", "embedding": tempemb, "orglist": temporgs}

with open("../../data/embeddings/Graph2Vec/C4_embedding.pkl", "wb") as f:
    pickle.dump(embtosave, f)
