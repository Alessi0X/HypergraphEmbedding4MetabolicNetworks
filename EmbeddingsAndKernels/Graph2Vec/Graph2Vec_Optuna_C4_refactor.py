import os
import sys
import optuna
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import hypergraphx as hgx
from hypergraphx.representations.projections import clique_projection
from tqdm import tqdm
from pyclustertend import hopkins

# Add parent directory to Python path for custom imports
sys.path.insert(0, os.path.join(os.path.dirname(__name__), "..", ".."))
from customgraph2vec.mygraph2vec import MyGraph2Vec

# Create folders to save Optuna studies
os.makedirs("../../data/optuna_studies", exist_ok=True)

# Set some useful parameters
ITERATION_NUMBER = 300  # Number of iterations for Optuna
nWorkers_Optuna = 1  # Number of workers for Optuna (i.e., number of parallel trials)
nWorkers_Graph2Vec = os.cpu_count()  # Number of workers for Graph2Vec

# Load hypergraph dataset and evaluate clique projections
with open("../../data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pickle.load(f)["DATASET"]

    for i in tqdm(range(len(DATASET)), desc="Evaluating clique projections"):
        tempnet = DATASET[i]["simplices_nodelabels"]
        orgname = DATASET[i]["ID"].upper()
        hnet = hgx.Hypergraph()
        hnet.add_edges(tempnet)
        cliquep = clique_projection(hnet, keep_isolated=False)
        DATASET[i] = {"ID": orgname, "Graph": cliquep}

# extract orglist and graphlist
orglist = []
graphlist = []
for i in range(len(DATASET)):
    orglist.append(DATASET[i]["ID"].upper())
    graphlist.append(DATASET[i]["Graph"])

prepgraphlist = []
for graph in tqdm(graphlist, desc="Preparing graphs for Graph2Vec"):
    for node in graph.nodes():
        graph.nodes[node]["feature"] = node
    mapping = {name: idx for idx, name in enumerate(graph.nodes())}
    temp = nx.relabel_nodes(graph, mapping)
    prepgraphlist.append(temp)

# load labels
labels = pd.read_csv("../../data/organismsLabels.csv")
labels["Organism_shortname"] = labels["Organism_shortname"].fillna("NAN")
labels["Organism_shortname"] = labels["Organism_shortname"].str.upper()
labels = labels[["Organism_shortname", "C4"]]

# select only labels with at least 20 occurrences
c4count = labels["C4"].value_counts().reset_index()
c4count.columns = ["C4", "count"]
c4toremove = c4count[c4count["count"] < 20]["C4"].tolist()
labels = labels[~(labels["C4"].isin(c4toremove))]

missinglist = labels.loc[(labels["C4"].isna()), "Organism_shortname"].tolist()

# remove labels not having a C4 label
labels = labels[~(labels["Organism_shortname"].isin(missinglist))]

# remove labels not in organisms
labels = labels[labels["Organism_shortname"].isin(orglist)]

orgsc4 = labels["Organism_shortname"].values

# arrange labels as orgsc4
labels["Organism_shortname"] = pd.Categorical(
    labels["Organism_shortname"], categories=orgsc4, ordered=True
)
print(f"C4 has {labels['C4'].nunique()} unique classes")

labels = labels.sort_values("Organism_shortname")

# cut prepgraphlist acordingly
prepgraphlistcut = []
orglist = np.array(orglist)
for org in orgsc4:
    idx = np.where(orglist == org)[0][0]
    prepgraphlistcut.append(prepgraphlist[idx])


def objective_function(trial):
    dimensions = trial.suggest_int("dimensions", 50, 300, step=1)
    wl = trial.suggest_int("wl_iterations", 1, 4, step=1)
    epochs = trial.suggest_int("epochs", 5, 40, step=1)
    lr = trial.suggest_float("lr", 0.01, 0.1, step=0.005)
    mincount = trial.suggest_int("mincount", 1, 1000, step=1)
    # print(
    #     f"parameters: dimensions={dimensions}, wl_iterations={wl}, epochs={epochs}, learning_rate={lr}, mincount={mincount}"
    # )
    # print("computing Graph2Vec")
    model = MyGraph2Vec(
        workers=nWorkers_Graph2Vec,
        dimensions=dimensions,
        wl_iterations=wl,
        epochs=epochs,
        learning_rate=lr,
        seed=472001,
        attributed=True,
        min_count=mincount,
    )

    model.fit(prepgraphlistcut)
    emb = model.get_embedding()

    # compute fitness value
    # print("computing hopkins...")
    hop = hopkins(emb, 360)

    return hop


# if the file exists, load it
# if os.path.exists("../../data/optuna_studies/NewGraph2VecParamOpt/graph2vecC4.pkl"):
#     with open(
#         "../../data/optuna_studies/NewGraph2VecParamOpt/graph2vecC4.pkl", "rb"
#     ) as f:
#         study = pickle.load(f)
#     best_homo = study.best_value
# else:
#     print("STARTING FROM SCRATCH!!!")
#     sampler = optuna.samplers.TPESampler(multivariate=True)
#     study = optuna.create_study(
#         direction="minimize", study_name="NewGraph2VecParamOpt", sampler=sampler
#     )
#
# print("PREPPED FOR STUDY")
#
# for i in range(ITERATION_NUMBER):
#     study.optimize(objective_function, n_trials=EVALS_PER_ITERATION, n_jobs=1)
#     best_hyperparameters = study.best_params
#     best_hop = study.best_value
#     print("LOOP:", i + 1)
#     print("Best Hyperparameters:", best_hyperparameters)
#     print("Best Hopikins:", best_hop)
#     # save opt for warmstart
#     with open("../../data/optuna_studies/Graph2VecParamOpt/graph2vecC4.pkl", "wb") as f:
#         pickle.dump(study, f)
#         print("SAVED!!")

# Setup
sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(
    direction="minimize", study_name="NewGraph2VecParamOpt", sampler=sampler
)

# Trigger
study.optimize(objective_function, n_trials=ITERATION_NUMBER, n_jobs=nWorkers_Optuna)
# best_hyperparameters = study.best_params
# best_hop = study.best_value

# Save
with open("../../data/optuna_studies/graph2vecC4.pkl", "wb") as f:
    pickle.dump(study, f)
    # print("SAVED!!")
