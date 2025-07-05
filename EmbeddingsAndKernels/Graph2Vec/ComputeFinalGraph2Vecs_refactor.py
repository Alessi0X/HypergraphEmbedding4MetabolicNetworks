import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from mynewbeautifulgraph2vec.mygraph2vec import MyGraph2Vec


# import clique projections
with open("data\\CliqueProjectionDataset.pkl", "rb") as f:
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
labels = pd.read_csv("data\\organismsLabels.csv")
labels["Organism_shortname"] = labels["Organism_shortname"].fillna("NAN")
labels["Organism_shortname"] = labels["Organism_shortname"].str.upper()

for clev in ["C1C2", "C3", "C4"]:
    templabels = labels[labels["Organism_shortname"].isin(orglist)].copy(deep=True)
    temporgs = orglist
    if clev in ["C3", "C4"]:
        if clev == "C3":
            minocc = 100
        else:  # clev == "C4"
            minocc = 20

        # select only labels with at least 100 occurrences
        count = labels[clev].value_counts().reset_index()
        count.columns = [clev, "count"]
        toremove = count[count["count"] < minocc][clev].tolist()
        templabels = templabels[~(templabels[clev].isin(toremove))]

        if clev == "C4":
            missinglist = templabels.loc[(templabels[clev].isna()), "Organism_shortname"].tolist()
            #remove organisms not having a C4 label
            templabels = templabels[~(templabels["Organism_shortname"].isin(missinglist))]

        temporgs = templabels["Organism_shortname"].values

    if clev == "C1C2":
        tempgraphlist = prepgraphlist
    else:
        #cut prepgraphlist acordingly
        tempgraphlist = []
        orglist = np.array(orglist)
        for org in temporgs:
            idx = np.where(orglist == org)[0][0]
            tempgraphlist.append(prepgraphlist[idx])
    if clev == "C1C2":
        for c in ["C1", "C2"]:
            print(f"for {c} there are {len(tempgraphlist)} eligible graphs,"
              f"{len(temporgs)} eligible organisms and"
              f" {templabels[c].nunique()} unique classes")
    else:
        print(f"for {clev} there are {len(tempgraphlist)} eligible graphs,"
          f"{len(temporgs)} eligible organisms and"
          f" {templabels[clev].nunique()} unique classes")

    with open(f"data/optuna_studies/NewGraph2VecParamOpt/graph2vec{clev}_hop.pkl", "rb") as f:
        tempstudy = pickle.load(f)
    tempparams = tempstudy.best_params

    print(f"best parameters for {clev} are: {tempparams}")

    tempg2v = MyGraph2Vec(workers = 1,
                      dimensions=tempparams["dimensions"],
                      wl_iterations=tempparams["wl_iterations"],
                      epochs=tempparams["epochs"],
                      learning_rate=tempparams["lr"],
                      seed=472001,
                      attributed=True,
                      min_count=tempparams["mincount"])
    print(f"computing Graph2Vec")

    tempg2v.fit(tempgraphlist)
    tempemb = tempg2v.get_embedding()

    print(f"embedding for class {clev} shape is: {tempemb.shape}")

    embtosave = {
        "classlev": clev,
        "ebedding": tempemb,
        "orglist": temporgs
                 }

    # save embedding
    with open(f"data/embeddings/NewGraph2Vec/{clev}_embedding.pkl", "wb") as f:
        pickle.dump(embtosave, f)

