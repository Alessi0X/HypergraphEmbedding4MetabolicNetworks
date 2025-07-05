# HypergraphEmbedding4MetabolicNetworks

This repository contains the code and data for the paper "Comparing the ability of embedding methods on metabolic hypergraphs for capturing taxonomy-based features".

The paper is currently under review on _Algorithms for Molecular Biology_.

<!-- ## Data
The dataset used in the paper is available in the file `MetabolicPathways_DATASET_Python.pkl`. This file contains the metabolic pathways data in a format suitable for analysis. -->

## Dependencies
To run the code, you need to install the following Python packages:
- list here

## Data
An example of the metabolic pathways dataset used in the paper is available in the file `example data/MetabolicPathways_DEMO_DATASET_Python.pkl`. This file contains the metabolic pathways data in a format suitable for analysis. This example dataset is a smaller version of the dataset used in the paper (5 organisms only), and it is intended for demonstration purposes only. The full list of organisms is available as a supplementary file in the paper.

The Pickle file contains a dictionary with `'DATASET'` as the key and a list of dictionaries as the value. Each dictionary in the list represents an organism and contains the following keys:
- `'ID'`: the ID of the organism
- `'simplices_nodelabels'`: the hyperedge list of the organism, where each hyperedge is represented as a tuple of node labels.