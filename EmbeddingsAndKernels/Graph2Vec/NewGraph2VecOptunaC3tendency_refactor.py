import optuna
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import networkx as nx
from mynewbeautifulgraph2vec.mygraph2vec import MyGraph2Vec
from pyclustertend import hopkins


ITERATION_NUMBER = 300
EVALS_PER_ITERATION = 1

# import clique projections
with open("data/CliqueProjectionDataset.pkl", "rb") as f:
    DATASET = pickle.load(f)["DATASET"]

#extract orglist and graphlist
orglist = []
graphlist = []
for i in range(len(DATASET)):
    orglist.append(DATASET[i]["ID"].upper())
    graphlist.append(DATASET[i]["Graph"])

prepgraphlist = []
for graph in tqdm(graphlist):
    for node in graph.nodes():
        graph.nodes[node]["feature"] = node
    mapping = {name: idx for idx, name in enumerate(graph.nodes())}
    temp = nx.relabel_nodes(graph, mapping)
    prepgraphlist.append(temp)

#load labels
labels = pd.read_csv("data/organismsLabels.csv")
labels["Organism_shortname"] = labels["Organism_shortname"].fillna("NAN")
labels["Organism_shortname"] = labels["Organism_shortname"].str.upper()
labels = labels[["Organism_shortname", "C3"]]

#select only labels with at least 100 occurrences
c3count = labels["C3"].value_counts().reset_index()
c3count.columns = ["C3", "count"]
c3toremove = c3count[c3count["count"] < 100]["C3"].tolist()
labels = labels[~(labels["C3"].isin(c3toremove))]

#remove labels not in organisms
labels = labels[labels["Organism_shortname"].isin(orglist)]

orgsc3 = labels["Organism_shortname"].values

#arrange labels as orgsc3
labels["Organism_shortname"] = pd.Categorical(labels["Organism_shortname"], categories=orgsc3, ordered=True)
print(f"C3 has {labels['C3'].nunique()} unique classes")

labels = labels.sort_values("Organism_shortname")

#cut prepgraphlist acordingly
prepgraphlistcut= []
orglist = np.array(orglist)
for org in orgsc3:
    idx = np.where(orglist == org)[0][0]
    prepgraphlistcut.append(prepgraphlist[idx])

def objective_function(trial):
    dimensions = trial.suggest_int("dimensions", 50, 300, step=1)
    wl = trial.suggest_int("wl_iterations", 1, 4, step=1)
    epochs = trial.suggest_int("epochs", 5, 40, step=1)
    lr = trial.suggest_float("lr", 0.01, 0.1, step=0.005)
    mincount = trial.suggest_int("mincount", 1, 1000, step=1)
    print(f"parameters: dimensions={dimensions}, wl_iterations={wl}, epochs={epochs}, learning_rate={lr}, mincount={mincount}")
    print("computing Graph2Vec")
    model = MyGraph2Vec(workers = 1,
                      dimensions=dimensions,
                      wl_iterations=wl,
                      epochs=epochs,
                      learning_rate=lr,
                      seed=472001,
                      attributed=True,
                      min_count=mincount)

    model.fit(prepgraphlistcut)
    emb = model.get_embedding()

    #compute entropy
    print("computing hopkins...")
    hop = hopkins(emb, 760)

    return hop



#if the file exists, load it
if os.path.exists('data/optuna_studies/NewGraph2VecParamOpt/graph2vecC3_hop.pkl'):
    with open('data/optuna_studies/NewGraph2VecParamOpt/graph2vecC3_hop.pkl', 'rb') as f:
        study = pickle.load(f)
    best_hop = study.best_value
else:
    print("STARTING FROM SCRATCH!!!")
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(direction="minimize", study_name="NewGraph2VecParamOpt", sampler=sampler)

print("PREPPED FOR STUDY")

for i in range(ITERATION_NUMBER):
    study.optimize(objective_function, n_trials=EVALS_PER_ITERATION, n_jobs=1)
    best_hyperparameters = study.best_params
    best_hop = study.best_value
    print("LOOP:", i+1)
    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Hopikins:", best_hop)
    # save opt for warmstart
    with open('data/optuna_studies/NewGraph2VecParamOpt/graph2vecC3_hop.pkl', 'wb') as f:
        pickle.dump(study, f)
        print("SAVED!!")

