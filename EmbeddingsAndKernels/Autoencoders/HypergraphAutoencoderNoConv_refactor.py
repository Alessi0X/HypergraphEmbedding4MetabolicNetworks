import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle as pkl
from tqdm import tqdm
import os

# ========================
# Data Loading and Preprocessing
# ========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
with open("data/MetabolicPathways_DATASET_Python.pkl", "rb") as f:
    DATA = pkl.load(f)
DATASET = DATA["DATASET"]

# Build global sets
nodeset = set()
hyperedgeset = set()
for graph in tqdm(DATASET, desc="Computing global node set"):
    hyperedges = graph["simplices_nodelabels"]
    for edge in hyperedges:
        edge_tuple = tuple(sorted(edge))
        hyperedgeset.add(edge_tuple)
        for node in edge:
            nodeset.add(node)

# Create mappings
global_node_to_idx = {node: i for i, node in enumerate(sorted(nodeset))}
global_edge_to_idx = {edge: i for i, edge in enumerate(sorted(hyperedgeset))}

def process_hypergraph(hyperedges, global_mapping_nodes, global_mapping_edges, dense=True):
    local_nodes = sorted(set(node for edge in hyperedges for node in edge))
    local_mapping = {node: i for i, node in enumerate(local_nodes)}

    row_indices = []
    col_indices = []
    for col, edge in enumerate(hyperedges):
        for node in edge:
            if node in local_mapping:
                row_indices.append(local_mapping[node])
                col_indices.append(col)

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.ones(len(row_indices), dtype=torch.float)
    shape = (len(local_nodes), len(hyperedges))

    if dense:
        incidence = torch.zeros(shape, dtype=torch.float)
        incidence[indices[0], indices[1]] = values
    else:
        incidence = torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float).coalesce()

    node_labels = torch.tensor([global_mapping_nodes[node] for node in local_nodes], dtype=torch.long)

    hyperedge_labels = []
    for edge in hyperedges:
        edge_tuple = tuple(sorted(edge))
        hyperedge_labels.append(global_mapping_edges[edge_tuple])
    hyperedge_labels = torch.tensor(hyperedge_labels, dtype=torch.long)

    return incidence, node_labels, hyperedge_labels

# Create hypergraph dataset
hypergraph_dataset = []
for graph in tqdm(DATASET, desc="Processing hypergraphs"):
    hyperedges = graph["simplices_nodelabels"]
    H, nodes, edge_labels = process_hypergraph(hyperedges, global_node_to_idx, global_edge_to_idx, dense=True)
    hypergraph_dataset.append((H, nodes, edge_labels))

# ========================
# Model Definition
# ========================

class HypergraphAutoencoder(nn.Module):
    def __init__(self, num_node_labels, num_edge_labels, embed_dim=32, conv_dim=64):
        super(HypergraphAutoencoder, self).__init__()
        self.embedding = nn.Embedding(num_node_labels, embed_dim)
        self.edge_embedding = nn.Embedding(num_edge_labels, embed_dim)


    def _summarize_embeddings(self, embeds):
        return embeds.mean(dim=0)

    def forward(self, node_labels, hyperedge_labels):
        node_embeds = self.embedding(node_labels)

        # Edge embeddings
        edge_embeds = self.edge_embedding(hyperedge_labels)  # shape: [num_edges, embed_dim]

        # Average edge embedding
        edge_rep = self._summarize_embeddings(edge_embeds)

        # Create combined vector j
        j = edge_rep

        # Reconstruction logits and sigmoid
        recon_logits = torch.matmul(node_embeds, edge_embeds.t())

        return recon_logits, j


def binary_agreement(recon: torch.Tensor, incidence: torch.Tensor, threshold: float = 0.5) -> float:
    recon_bin = (recon > threshold).int()
    incidence_bin = (incidence > threshold).int()
    nonzeros_incidence = torch.sum(incidence_bin != 0)
    intersection = torch.sum((recon_bin != 0) & (incidence_bin != 0))
    hamming_distance = nonzeros_incidence - intersection
    agreement = 1 - (hamming_distance.float() / nonzeros_incidence.float())
    return agreement.item()

def binary_f1_score(recon: torch.Tensor, incidence: torch.Tensor, threshold: float = 0.5) -> float:
    recon_bin = (recon > threshold).int()
    incidence_bin = (incidence > threshold).int()
    tp = torch.sum((recon_bin == 1) & (incidence_bin == 1))
    predicted_positives = torch.sum(recon_bin == 1)
    actual_positives = torch.sum(incidence_bin == 1)
    if predicted_positives == 0 or actual_positives == 0:
        return 0.0
    precision = tp.float() / predicted_positives.float()
    recall = tp.float() / actual_positives.float()
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1.item()

# ========================
# Training Setup
# ========================

model = HypergraphAutoencoder(
    num_node_labels=len(global_node_to_idx),
    num_edge_labels=len(global_edge_to_idx),
    embed_dim=400,
    conv_dim=400
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

numepochs = 13
os.makedirs("data/TorchModelCheckpoints/AutoencoderV4", exist_ok=True)

for epoch in range(numepochs):
    total_loss = 0.0
    agre = 0.0
    f1total = 0.0
    model.train()
    for incidence, node_labels, hyperedge_labels in tqdm(hypergraph_dataset, desc=f"Epoch {epoch + 1}/{numepochs}"):
        incidence = incidence.to(device)
        node_labels = node_labels.to(device)
        hyperedge_labels = hyperedge_labels.to(device)
        optimizer.zero_grad()
        recon_logit, j = model(node_labels, hyperedge_labels)
        loss = loss_fn(recon_logit, incidence)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon = torch.sigmoid(recon_logit)
        agre += binary_agreement(recon, incidence)
        f1total += binary_f1_score(recon, incidence)

    avg_agre = agre / len(hypergraph_dataset)
    avg_loss = total_loss / len(hypergraph_dataset)
    avg_f1 = f1total / len(hypergraph_dataset)

    if epoch % 3 == 0:
        checkpoint_path = f"data/TorchModelCheckpoints/AutoencoderV4/Checkpoint_Epoch_{epoch}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_loss,
            'avg_agre': avg_agre,
            'avg_f1': avg_f1
        }
        torch.save(checkpoint, checkpoint_path)

    with open("data/TorchModelCheckpoints/AutoencoderV4/performances.txt", "a") as f:
        f.write(f"Epoch {epoch + 1}: Average Loss = {avg_loss}, Avg. Agreement = {avg_agre}, Avg. F1 = {avg_f1}\n")

    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Avg. Agreement = {avg_agre:.4f}, Avg. F1 = {avg_f1:.4f}")

# ========================
# Extracting the Global Latent Features (z)
# ========================

model = HypergraphAutoencoder(num_node_labels=len(global_node_to_idx),
                              num_edge_labels=len(global_edge_to_idx),
                              embed_dim=400,
                              conv_dim=400).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

checkpoint = torch.load("data/TorchModelCheckpoints/AutoencoderV4/Checkpoint_Epoch_12.pth")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()
all_z = []
with torch.no_grad():
    for incidence, node_labels, hyperedge_labels in tqdm(hypergraph_dataset):
        incidence = incidence.to(device)
        node_labels = node_labels.to(device)
        hyperedge_labels = hyperedge_labels.to(device)
        _, j = model(node_labels, hyperedge_labels)
        #recon_cpu = _.cpu().numpy()
        all_z.append(j.cpu().numpy())

# Stack the latent vectors into a feature matrix of shape [num_hypergraphs, bottleneck_dim]
feature_matrix = np.stack(all_z)

#save feature mat
with open("data/TorchModelCheckpoints/AutoencoderV4/feature_matrix.pkl", "wb") as f:
    pkl.dump(feature_matrix, f)

#save organism labels
orglabels = [graph["ID"].upper() for graph in DATASET]

with open("data/TorchModelCheckpoints/AutoencoderV4/orglabels.pkl", "wb") as f:
    pkl.dump(orglabels, f)


# -------- compute distances and save with also org labels --------

from scipy.spatial.distance import pdist, squareform

# Compute pairwise distances
pairwise_distances = pdist(feature_matrix, metric='euclidean')

# Convert to square form
distance_matrix = squareform(pairwise_distances)

# Save the distance matrix
with open("data/distances/AutoEncoderV4NoConv.pkl", "wb") as f:
    pkl.dump(distance_matrix, f)
with open("data/distances/ORG_AutoEncoderV4NoConv.pkl", "wb") as f:
    pkl.dump(orglabels, f)
