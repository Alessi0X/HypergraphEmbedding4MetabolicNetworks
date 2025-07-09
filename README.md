# HypergraphEmbedding4MetabolicNetworks

This repository contains the code and data for the paper "Comparing the ability of embedding methods on metabolic hypergraphs for capturing taxonomy-based features".

The paper is currently under review on _Algorithms for Molecular Biology_.

## Usage
TBC

**Note**: to run the two Edit Kernels it is strongly recommended to compile the core script `levenshteinDistance.pyx` using Cython. This will significantly speed up the computation of the Edit Kernels. To compile the script, create a file named `setup.py` with the following content:

```python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("levenshteinDistance.pyx", annotate=True, compiler_directives={'language_level' : "3"})
)
```

Then, run the following command in the terminal:

```bash
python setup.py build_ext --inplace
```

## Requirements
To run the code, you need to install the following Python packages:
- torch 2.7.1
- torch_geometric 2.6.1
- hypergraphx 1.7.7
- karateclub 1.2.1
- networkx 3.5
- scipy 1.15.3

## Data
An example of the metabolic pathways dataset used in the paper is available in the file `example data/MetabolicPathways_DEMO_DATASET_Python.pkl`. This file contains the metabolic pathways data in a format suitable for analysis. This example dataset is a smaller version of the dataset used in the paper (5 organisms only), and it is intended for demonstration purposes only. The full list of organisms is available as a supplementary file in the paper.

The Pickle file contains a dictionary with `'DATASET'` as the key and a list of dictionaries as the value. Each dictionary in the list represents an organism and contains the following keys:
- `'ID'`: the ID of the organism
- `'simplices_nodelabels'`: the hyperedge list of the organism, where each hyperedge is represented as a _n_-tuple of node labels (strings), with _n_ being the number of nodes in the hyperedge.

## Citation
If you use this code or data in your research, please cite the paper as follows:

TBC