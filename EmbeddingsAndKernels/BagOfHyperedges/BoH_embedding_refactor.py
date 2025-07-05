import pandas as pd
import time
from tqdm import tqdm
import pickle as pkl

start = time.time()

#load dataset
with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATASET = pkl.load(f)["DATASET"]

numGraphs = len(DATASET)

# BAG OF WORDS EMBEDDING

# get set of organisms
organisms = [data['ID'] for data in DATASET]

# Create embeddings with a progress bar
bagofwords_dict = {}

for idx in tqdm(range(numGraphs), desc="Processing Files"):
    content = DATASET[idx]["simplices_nodelabels"]
    org = DATASET[idx]["ID"].upper()
    content = [",".join(sorted(x)) for x in content]
    for hyperedge in content:
        bagofwords_dict.setdefault(org, {})[hyperedge] = 1

# Convert dict to DataFrame
embeddingdf = pd.DataFrame.from_dict(bagofwords_dict, orient="index").fillna(0)
embeddingcopy = embeddingdf.copy(deep=True)

#check if embedding df has the same elements as organisms
organisms = [x.upper() for x in organisms]
for org in organisms:
    if org not in embeddingdf.index:
        print(org)

#sort the embedding df acording to organisms
embeddingdf = embeddingdf.loc[organisms]

# save created embedding
print("Saving embedding with shape: ", embeddingdf.shape)
embeddingdf.to_csv("data/embeddings/BagOfWords.csv")

#create distance matrices
print("Calculating distance matrices")
from scipy.spatial.distance import pdist, squareform

# extract
embeddingmatrix = embeddingdf.values

# calculate distance matrix

distmatrix = pdist(embeddingmatrix, metric="jaccard")
distmatrix = squareform(distmatrix)

# save distance matrix
with open("data/distances/BagOfWordsJaccard.pkl", "wb") as f:
    pkl.dump(distmatrix, f)
with open("data/distances/ORG_BagOfWordsJaccard.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)
print("Jaccard distance matrix saved, time: ", time.time() - start)

distmatrixman = pdist(embeddingmatrix, metric="cityblock")
distmatrixman = squareform(distmatrixman)

# save distance matrix
with open("data/distances/BagOfWordsManhattan.pkl", "wb") as f:
    pkl.dump(distmatrixman, f)
with open("data/distances/ORG_BagOfWordsManhattan.pkl", "wb") as f:
    pkl.dump(embeddingdf.index.tolist(), f)

print(f"time elapsed: {time.time() - start}")