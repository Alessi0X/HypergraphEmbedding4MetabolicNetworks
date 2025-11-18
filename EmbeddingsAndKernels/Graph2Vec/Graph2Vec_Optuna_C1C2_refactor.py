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
with open("../../data/MetabolicPathways_DEMO_DATASET_Python.pkl", "rb") as f:
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

    model.fit(prepgraphlist)
    emb = model.get_embedding()

    # compute fitness value
    # print("computing hopkins...")
    hop = hopkins(emb, 850)

    return hop


# if the file exists, load it
# if os.path.exists("../../data/optuna_studies/NewGraph2VecParamOpt/graph2vecC1C2.pkl"):
#     with open(
#         "../../data/optuna_studies/NewGraph2VecParamOpt/graph2vecC1C2.pkl", "rb"
#     ) as f:
#         study = pickle.load(f)
#     best_hop = study.best_value
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
#     print("Best Hopkins:", best_hop)
#     # save opt for warmstart
#     with open(
#         "../../data/optuna_studies/NewGraph2VecParamOpt/graph2vecC1C2.pkl", "wb"
#     ) as f:
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
with open("../../data/optuna_studies/graph2vecC1C2.pkl", "wb") as f:
    pickle.dump(study, f)
    # print("SAVED!!")
